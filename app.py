import copy
import json
import os
import logging
import re
import uuid
import httpx
import asyncio
from quart import (
    Blueprint,
    Quart,
    jsonify,
    make_response,
    request,
    send_from_directory,
    render_template,
    current_app,
)

from openai import AsyncAzureOpenAI
from azure.identity.aio import (
    DefaultAzureCredential,
    get_bearer_token_provider
)
from backend.auth.auth_utils import get_authenticated_user_details
from backend.security.ms_defender_utils import get_msdefender_user_json
from backend.history.cosmosdbservice import CosmosConversationClient
from backend.settings import (
    app_settings,
    MINIMUM_SUPPORTED_AZURE_OPENAI_PREVIEW_API_VERSION
)
from backend.utils import (
    format_as_ndjson,
    format_stream_response,
    format_non_streaming_response,
    convert_to_pf_format,
    format_pf_non_streaming_response,
)

import time
import datetime
try:
    from backend.permit_agent import agent as permit_agent
except Exception:  # missing aiohttp etc. -> feature simply stays off
    permit_agent = None
    logging.exception("permit agent unavailable; permit questions fall back to RAG")
try:
    from backend.permit_agent import apply_agent
except Exception:  # missing deps -> apply flow simply stays off
    apply_agent = None
    logging.exception("apply agent unavailable; instant-permit flow disabled")
try:  # mock conversational flows (public-record request + inspection scheduling)
    from backend.permit_agent import prr_agent, inspection_agent
except Exception:
    prr_agent = inspection_agent = None
    logging.exception("mock flow agents unavailable; PRR + inspection disabled")
try:
    from backend import meetings as meetings_feed
except Exception:  # missing deps -> live meeting lookup stays off
    meetings_feed = None
    logging.exception("meetings feed unavailable; meeting questions fall back to RAG")

bp = Blueprint("routes", __name__, static_folder="static", template_folder="static")

cosmos_db_ready = asyncio.Event()


def create_app():
    app = Quart(__name__)
    app.register_blueprint(bp)
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    
    @app.before_serving
    async def init():
        try:
            app.cosmos_conversation_client = await init_cosmosdb_client()
            cosmos_db_ready.set()
        except Exception as e:
            logging.exception("Failed to initialize CosmosDB client")
            app.cosmos_conversation_client = None
            raise e
    
    return app


@bp.route("/")
async def index():
    return await render_template(
        "index.html",
        title=app_settings.ui.title,
        favicon=app_settings.ui.favicon
    )


@bp.route("/favicon.ico")
async def favicon():
    return await bp.send_static_file("favicon.ico")


@bp.route("/assets/<path:path>")
async def assets(path):
    return await send_from_directory("static/assets", path)


# Debug settings
DEBUG = os.environ.get("DEBUG", "false")
log_level = logging.DEBUG if DEBUG.lower() == "true" else logging.INFO
# force=True replaces gunicorn's root logger config (which defaults to WARNING
# and would otherwise silence our INFO-level [USER QUERY]/[RETRIEVED CHUNKS] logs)
logging.basicConfig(level=log_level, force=True)
# Azure SDKs log every HTTP request/response at INFO; silence that noise
logging.getLogger("azure").setLevel(logging.WARNING)

USER_AGENT = "GitHubSampleWebApp/AsyncAzureOpenAI/1.0.0"


# Frontend Settings via Environment Variables
frontend_settings = {
    "auth_enabled": app_settings.base_settings.auth_enabled,
    "feedback_enabled": (
        app_settings.chat_history and
        app_settings.chat_history.enable_feedback
    ),
    "ui": {
        "title": app_settings.ui.title,
        "logo": app_settings.ui.logo,
        "chat_logo": app_settings.ui.chat_logo or app_settings.ui.logo,
        "chat_title": app_settings.ui.chat_title,
        "chat_description": app_settings.ui.chat_description,
        "show_share_button": app_settings.ui.show_share_button,
        "show_chat_history_button": app_settings.ui.show_chat_history_button,
        "chat_response_contactmessage": app_settings.ui.chat_response_contactmessage,
        "poweredby": app_settings.ui.poweredby,
        "poweredbycomp": app_settings.ui.poweredbycomp,
        "poweredbyurl": app_settings.ui.poweredbyurl,
        "headertitle": app_settings.ui.headertitle,
        "example_title": app_settings.ui.example_title,
        "example_option_1": app_settings.ui.example_option_1,
        "example_option_2":  app_settings.ui.example_option_2,
        "example_option_3":  app_settings.ui.example_option_3,
        "example_option_4":  app_settings.ui.example_option_4,
        "capabilities":  app_settings.ui.capabilities,
        "capabilities_1":  app_settings.ui.capabilities_1,
        "capabilities_2": app_settings.ui.capabilities_2,
        "capabilities_3": app_settings.ui.capabilities_3,
        "limitations": app_settings.ui.limitations,
        "limitations_1": app_settings.ui.limitations_1,
        "limitations_2": app_settings.ui.limitations_2,
        "limitations_3": app_settings.ui.limitations_3,
        "chat_resp_logo": app_settings.ui.chat_resp_logo,
        "hand_wave_icon": app_settings.ui.hand_wave_icon,
        "show_permit_link": app_settings.ui.show_permit_link
    },
    "sanitize_answer": app_settings.base_settings.sanitize_answer,
    "oyd_enabled": app_settings.base_settings.datasource_type,
}


# Enable Microsoft Defender for Cloud Integration
MS_DEFENDER_ENABLED = os.environ.get("MS_DEFENDER_ENABLED", "true").lower() == "true"


# Initialize Azure OpenAI Client
async def init_openai_client():
    azure_openai_client = None
    
    try:
        # API version check
        if (
            app_settings.azure_openai.preview_api_version
            < MINIMUM_SUPPORTED_AZURE_OPENAI_PREVIEW_API_VERSION
        ):
            raise ValueError(
                f"The minimum supported Azure OpenAI preview API version is '{MINIMUM_SUPPORTED_AZURE_OPENAI_PREVIEW_API_VERSION}'"
            )

        # Endpoint
        if (
            not app_settings.azure_openai.endpoint and
            not app_settings.azure_openai.resource
        ):
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_RESOURCE is required"
            )

        endpoint = (
            app_settings.azure_openai.endpoint
            if app_settings.azure_openai.endpoint
            else f"https://{app_settings.azure_openai.resource}.openai.azure.com/"
        )

        # Authentication
        aoai_api_key = app_settings.azure_openai.key
        ad_token_provider = None
        if not aoai_api_key:
            logging.debug("No AZURE_OPENAI_KEY found, using Azure Entra ID auth")
            async with DefaultAzureCredential() as credential:
                ad_token_provider = get_bearer_token_provider(
                    credential,
                    "https://cognitiveservices.azure.com/.default"
                )

        # Deployment
        deployment = app_settings.azure_openai.model
        if not deployment:
            raise ValueError("AZURE_OPENAI_MODEL is required")

        # Default Headers
        default_headers = {"x-ms-useragent": USER_AGENT}

        azure_openai_client = AsyncAzureOpenAI(
            api_version=app_settings.azure_openai.preview_api_version,
            api_key=aoai_api_key,
            azure_ad_token_provider=ad_token_provider,
            default_headers=default_headers,
            azure_endpoint=endpoint,
        )

        return azure_openai_client
    except Exception as e:
        logging.exception("Exception in Azure OpenAI initialization", e)
        azure_openai_client = None
        raise e


async def init_cosmosdb_client():
    cosmos_conversation_client = None
    if app_settings.chat_history:
        try:
            cosmos_endpoint = (
                f"https://{app_settings.chat_history.account}.documents.azure.com:443/"
            )

            if not app_settings.chat_history.account_key:
                async with DefaultAzureCredential() as cred:
                    credential = cred
                    
            else:
                credential = app_settings.chat_history.account_key

            cosmos_conversation_client = CosmosConversationClient(
                cosmosdb_endpoint=cosmos_endpoint,
                credential=credential,
                database_name=app_settings.chat_history.database,
                container_name=app_settings.chat_history.conversations_container,
                enable_message_feedback=app_settings.chat_history.enable_feedback,
            )
        except Exception as e:
            logging.exception("Exception in CosmosDB initialization", e)
            cosmos_conversation_client = None
            raise e
    else:
        logging.debug("CosmosDB not configured")

    return cosmos_conversation_client


def prepare_model_args(request_body, request_headers):
    request_messages = request_body.get("messages", [])
    messages = []
    if not app_settings.datasource:
        messages = [
            {
                "role": "system",
                "content": f"Today's date is {datetime.date.today().isoformat()}. "
                + app_settings.azure_openai.system_message
            }
        ]

    for message in request_messages:
        if message:
            if message["role"] == "assistant" and "context" in message:
                context_obj = json.loads(message["context"])
                messages.append(
                    {
                        "role": message["role"],
                        "content": message["content"],
                        "context": context_obj
                    }
                )
            else:
                messages.append(
                    {
                        "role": message["role"],
                        "content": message["content"]
                    }
                )

    user_json = None
    if (MS_DEFENDER_ENABLED):
        authenticated_user_details = get_authenticated_user_details(request_headers)
        conversation_id = request_body.get("conversation_id", None)
        application_name = app_settings.ui.title
        user_json = get_msdefender_user_json(authenticated_user_details, request_headers, conversation_id, application_name)

    model_args = {
        "messages": messages,
        "temperature": app_settings.azure_openai.temperature,
        "max_tokens": app_settings.azure_openai.max_tokens,
        "top_p": app_settings.azure_openai.top_p,
        "stop": app_settings.azure_openai.stop_sequence,
        "stream": app_settings.azure_openai.stream,
        "model": app_settings.azure_openai.model,
        "user": user_json
    }

    if app_settings.datasource:
        model_args["extra_body"] = {
            "data_sources": [
                app_settings.datasource.construct_payload_configuration(
                    request=request
                )
            ]
        }
        # Stamp today's date into the system prompt (role_information) so the model can
        # reason about "next / upcoming / recent / this year" instead of treating an old
        # indexed date as current. The env-var system message stays the static base text.
        _params = model_args["extra_body"]["data_sources"][0].get("parameters", {})
        for _k in ("role_information", "roleInformation"):
            if _params.get(_k):
                _params[_k] = f"Today's date is {datetime.date.today().isoformat()}. " + _params[_k]

    model_args_clean = copy.deepcopy(model_args)
    if model_args_clean.get("extra_body"):
        secret_params = [
            "key",
            "connection_string",
            "embedding_key",
            "encoded_api_key",
            "api_key",
        ]
        for secret_param in secret_params:
            if model_args_clean["extra_body"]["data_sources"][0]["parameters"].get(
                secret_param
            ):
                model_args_clean["extra_body"]["data_sources"][0]["parameters"][
                    secret_param
                ] = "*****"
        authentication = model_args_clean["extra_body"]["data_sources"][0][
            "parameters"
        ].get("authentication", {})
        for field in authentication:
            if field in secret_params:
                model_args_clean["extra_body"]["data_sources"][0]["parameters"][
                    "authentication"
                ][field] = "*****"
        embeddingDependency = model_args_clean["extra_body"]["data_sources"][0][
            "parameters"
        ].get("embedding_dependency", {})
        if "authentication" in embeddingDependency:
            for field in embeddingDependency["authentication"]:
                if field in secret_params:
                    model_args_clean["extra_body"]["data_sources"][0]["parameters"][
                        "embedding_dependency"
                    ]["authentication"][field] = "*****"

    user_query = next((m["content"] for m in reversed(model_args_clean["messages"]) if m["role"] == "user"), None)
    logging.info(f"[USER QUERY] {user_query}")

    return model_args


async def promptflow_request(request):
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {app_settings.promptflow.api_key}",
        }
        # Adding timeout for scenarios where response takes longer to come back
        logging.debug(f"Setting timeout to {app_settings.promptflow.response_timeout}")
        async with httpx.AsyncClient(
            timeout=float(app_settings.promptflow.response_timeout)
        ) as client:
            pf_formatted_obj = convert_to_pf_format(
                request,
                app_settings.promptflow.request_field_name,
                app_settings.promptflow.response_field_name
            )
            # NOTE: This only support question and chat_history parameters
            # If you need to add more parameters, you need to modify the request body
            response = await client.post(
                app_settings.promptflow.endpoint,
                json={
                    app_settings.promptflow.request_field_name: pf_formatted_obj[-1]["inputs"][app_settings.promptflow.request_field_name],
                    "chat_history": pf_formatted_obj[:-1],
                },
                headers=headers,
            )
        resp = response.json()
        resp["id"] = request["messages"][-1]["id"]
        return resp
    except Exception as e:
        logging.error(f"An error occurred while making promptflow_request: {e}")


# --- Domain routing: send each question to the right scoped index ----------------
# website (people/officials/contacts/services/FAQs) stays on the default index;
# permit and codes questions are rerouted to their dedicated indexes so the dense
# municipal-code text can't drown out, or be drowned by, the other domains.
PERMITS_INDEX = os.environ.get("AZURE_SEARCH_INDEX_PERMITS")
CODES_INDEX = os.environ.get("AZURE_SEARCH_INDEX_CODES")
# Routing is OPT-IN: active only when BOTH domain indexes are configured. Apps
# without these env vars (other cities, or prod until it's ready) behave exactly
# as before, no classifier call, no rerouting.
INDEX_ROUTING_ENABLED = bool(PERMITS_INDEX and CODES_INDEX)

ROUTER_SYSTEM_MESSAGE = (
    "You route a resident's question for a city government assistant to ONE data source. "
    "Reply with exactly one lowercase word: website, permit, instant-permit, public-record, "
    "inspection, or codes.\n"
    "- website: people, officials, departments, contacts, phone/email, hours, addresses, "
    "city services, news, events, FAQs, general how-to questions, AND how to apply for or "
    "pay for a permit, permit fees, what documents are needed, which permit you need for a "
    "project, what permit types the city offers in general, and Building & Safety info.\n"
    "- instant-permit: the user wants to ACTUALLY APPLY for / start / file / submit a permit "
    "application right now (e.g. 'I want to apply for a solar permit', 'help me file a solar "
    "permit', 'start my permit application'). ALSO classify as instant-permit when the resident "
    "reports that one of these is broken, failing, at end of life, or needs replacing, because "
    "the assistant can file the replacement permit: water heater, furnace or air "
    "conditioner/HVAC, electrical panel, house wiring (rewire), or water piping (repipe). "
    "Examples: 'my water heater is broken', 'I need to replace my electrical panel', 'my "
    "furnace died', 'time to repipe the house'. This is the transactional apply flow, NOT a "
    "how-to question (website) and NOT looking up existing records (permit).\n"
    "- public-record: the user wants to FILE / submit / start a public records request (CPRA). "
    "The abbreviation 'PRR' means public records request. Examples: 'I want to apply for a PRR', "
    "'submit a public records request', 'I need copies of city records'. This is a transactional "
    "request flow, NOT a permit.\n"
    "- inspection: the user wants to SCHEDULE, book, reschedule, or move a building INSPECTION "
    "for a permit. Examples: 'schedule an inspection', 'reschedule my final inspection', 'move my "
    "inspection to Thursday'. NOT a question about which inspections are required (that is website).\n"
    "- permit: looking up SPECIFIC existing permit records, their status, or any COUNT, "
    "BREAKDOWN, LIST, or RANKING of permits actually filed or issued (this also covers "
    "business tax registrations and business licenses). Includes breakdowns by type, "
    "status, or department, totals over a time period (a year/month), and 'which type or "
    "department has the most'. Examples: 'permit history for 150 N Third St', 'how many "
    "solar permits in December', 'breakdown of permit types in 2025', 'which department "
    "issued the most permits this year', 'how many new businesses opened in 2025'. Use for "
    "existing permit records and their aggregates, NOT for how to apply, fees, or what "
    "permit types exist in general.\n"
    "- codes: the municipal code text, ordinances, or regulations themselves (zoning, "
    "setbacks, what the code/law says).\n"
    "If you are unsure, answer website."
)


async def classify_domain(user_query, client, history=None):
    """Return 'website' | 'permit' | 'codes' for a question. Defaults to website on any failure.

    If `history` (recent user/assistant turns, ending with the current question) is given,
    the classifier sees it so a short follow-up like 'at what locations?' inherits the topic
    of the previous turn instead of being misread as a generic website question."""
    if not user_query:
        return "website"
    if history:
        system = ROUTER_SYSTEM_MESSAGE + (
            "\nThis is a multi-turn chat. Classify the user's MOST RECENT message, using the "
            "earlier turns only as context. A short follow-up ('at what locations?', 'and in "
            "2024?', 'what about commercial?') inherits the topic of the previous question.")
        convo = [{"role": "system", "content": system}] + history
    else:
        convo = [{"role": "system", "content": ROUTER_SYSTEM_MESSAGE},
                 {"role": "user", "content": user_query}]
    try:
        resp = await client.chat.completions.create(
            model=app_settings.azure_openai.model,
            messages=convo,
            temperature=0,
            max_tokens=5,
        )
        label = (resp.choices[0].message.content or "").strip().lower()
    except Exception:
        logging.exception("Domain classifier failed; defaulting to website")
        return "website"
    if "instant" in label:          # check before "permit" ('instant-permit' contains it)
        return "instant-permit"
    if "inspect" in label:
        return "inspection"
    if "record" in label or "public" in label:
        return "public-record"
    if "permit" in label:
        return "permit"
    if "code" in label:
        return "codes"
    return "website"


def index_for_domain(domain):
    """Map an ALREADY-classified domain to a scoped index name, or None to keep the default
    (website). No LLM call here; the domain was classified once via _domain_for and reused.

    When the permit AGENT is on, permit questions are handled by it (live records), not a
    RAG index, so only codes reroutes here.
    """
    if domain == "permit" and not PERMIT_AGENT_ENABLED:
        return PERMITS_INDEX
    if domain == "codes":
        return CODES_INDEX
    return None


# --- Permit agent: answer existing-permit questions from the live records ---------
# Opt-in (askburbanktest only for now). When on, a question the classifier tags as
# 'permit' is answered by the read-permits agent (counts/lists/lookups over the permits
# index) instead of RAG. Everything else (website, codes) is unchanged.
PERMIT_AGENT_ENABLED = bool(permit_agent) and os.environ.get("PERMIT_AGENT_ENABLED", "0") != "0"


def _latest_user_query(messages):
    return next(
        (m["content"] for m in reversed(messages)
         if m.get("role") == "user" and isinstance(m.get("content"), str)),
        None,
    )


def _recent_history(messages, turns=6, max_chars=700):
    """Last few user/assistant turns (content trimmed), so the classifier and the agent
    have conversation context for follow-up questions. Ends with the current question."""
    recent = [m for m in messages
              if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)]
    return [{"role": m["role"], "content": m["content"][:max_chars]} for m in recent[-turns:]]


async def _domain_for(request_body, client):
    """Classify the latest question ONCE per request and cache it on request_body.

    The apply, permit, and index-routing deciders all ask the same 'what domain is this?'
    question, so we run the classifier LLM once and every consumer reuses the cached word.
    (Mid-flow apply turns short-circuit before this, so they classify zero times.)"""
    if "_domain" in request_body:
        return request_body["_domain"]
    raw = request_body.get("messages", [])
    user_query = _latest_user_query(raw)
    domain = "website"
    if user_query:
        domain = await classify_domain(user_query, client, history=_recent_history(raw))
    request_body["_domain"] = domain
    return domain


async def try_permit_answer(request_body):
    """If the latest question is a permit-records question, answer it from the live
    permits index and return the answer string. Otherwise return None (run normal RAG)."""
    if not PERMIT_AGENT_ENABLED:
        return None
    messages = [m for m in request_body.get("messages", []) if m.get("role") != "tool"]
    user_query = _latest_user_query(messages)
    if not user_query:
        return None
    try:
        client = await init_openai_client()
        history = _recent_history(messages)
        if await _domain_for(request_body, client) != "permit":
            return None
        logging.info("[PERMIT AGENT] handling: %s", user_query)
        return await permit_agent.answer_permit_query(
            user_query, client, app_settings.azure_openai.model, history=history)
    except Exception:
        logging.exception("permit agent failed; falling back to RAG")
        return None


# Live meeting-schedule lookup (Burbank-specific: only active when its Granicus feed URL is set).
MEETINGS_ENABLED = bool(meetings_feed) and bool(os.environ.get("MEETINGS_FEED_URL"))


async def try_meetings_answer(request_body):
    """Answer 'when is the next <body> meeting' from the live Granicus feed. Returns the answer
    string, or None to fall through to RAG. Off unless MEETINGS_FEED_URL is set."""
    if not MEETINGS_ENABLED:
        return None
    messages = [m for m in request_body.get("messages", []) if m.get("role") != "tool"]
    user_query = _latest_user_query(messages)
    if not user_query or not meetings_feed.is_meeting_query(user_query):
        return None
    try:
        answer = await meetings_feed.answer_meeting_query(user_query, datetime.date.today())
        if answer:
            logging.info("[MEETINGS FEED] handled: %s", user_query)
        return answer
    except Exception:
        logging.exception("meetings feed failed; falling back to RAG")
        return None


def _permit_message_obj():
    return {
        "id": "permit-agent",
        "model": app_settings.azure_openai.model,
        "created": int(time.time()),
        "object": "extensions.chat.completion",
        "choices": [{"messages": []}],
    }


def permit_non_streaming_response(answer, history_metadata):
    """Shape a permit answer exactly like format_non_streaming_response output."""
    obj = _permit_message_obj()
    obj["choices"][0]["messages"].append({"role": "assistant", "content": answer})
    obj["history_metadata"] = history_metadata
    obj["apim-request-id"] = "permit-agent"
    return obj


def permit_stream_response(answer, history_metadata):
    """A one-chunk async stream shaped like format_stream_response output."""
    async def generate():
        obj = _permit_message_obj()
        obj["object"] = "extensions.chat.completion.chunk"
        obj["choices"][0]["messages"].append({"role": "assistant", "content": answer})
        obj["history_metadata"] = history_metadata
        obj["apim-request-id"] = "permit-agent"
        yield obj
    return generate()


# --- Instant-permit APPLY flow (multi-turn, in the chat UI) -----------------------
# Opt-in. State lives entirely in the conversation history (the apply LLM re-reads the
# collected fields each turn). A hidden tag in the assistant turn's `context` marks that
# we're mid-flow, so cheap code can skip the classifier while an application is in progress.
PERMIT_APPLY_ENABLED = bool(apply_agent) and os.environ.get("PERMIT_APPLY_ENABLED", "0") != "0"
APPLY_FLOW = "instant-permit"        # mid-flow tag; rides in a `context` -> `tool` message
APPLY_OFFER = "instant-permit-offer"  # pre-flow "want help applying?" yes/no offer tag
APPLY_PAY = "instant-permit-pay"      # post-submit: the pay card is showing, awaiting payment

# Mock conversational flows (public-record request + inspection). Both are fully mock: no RAG,
# no external API. Keyword-triggered, then they reuse the same tag/widget round-trip as apply.
MOCK_FLOWS_ENABLED = bool(prr_agent) and bool(inspection_agent) and os.environ.get("MOCK_FLOWS_ENABLED", "0") != "0"
PRR_OFFER = "public-record-offer"     # "want help submitting a request?" yes/no offer
PRR_FLOW = "public-record"            # mid-flow: collecting + reviewing the request
INSPECTION_FLOW = "inspection"        # mid-flow: scheduling/rescheduling


def _flow_family(tag):
    """Which flow owns this tag, so each try_* handler yields to the right one."""
    if tag in (APPLY_FLOW, APPLY_OFFER, APPLY_PAY):
        return "apply"
    if tag in (PRR_FLOW, PRR_OFFER):
        return "prr"
    if tag == INSPECTION_FLOW:
        return "inspection"
    return None


def _last_bot_flow(messages):
    """The `flow` value tagged on the MOST RECENT bot turn, or None. The frontend replays the
    tag as a `tool` message sitting just before the assistant message, so we find the last
    assistant message and read the tool message right before it. Only the last turn counts,
    so once the flow ends (no tag emitted) we correctly fall out."""
    idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            idx = i
            break
    if idx is None:
        return None
    prev = messages[idx - 1] if idx > 0 else None
    if not prev or prev.get("role") != "tool":
        return None
    try:
        return json.loads(prev.get("content") or "{}").get("flow")
    except (ValueError, TypeError):
        return None


def _apply_flow_active(messages):
    return _last_bot_flow(messages) == APPLY_FLOW


def _apply_offer_pending(messages):
    return _last_bot_flow(messages) == APPLY_OFFER


def _apply_pay_pending(messages):
    return _last_bot_flow(messages) == APPLY_PAY


def _last_bot_widget(messages):
    """The `widget` payload tagged on the most recent bot turn, or None. Used to recover the
    permit number + email from the pay card when the user pays."""
    idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            idx = i
            break
    if idx is None or idx == 0 or messages[idx - 1].get("role") != "tool":
        return None
    try:
        return json.loads(messages[idx - 1].get("content") or "{}").get("widget")
    except (ValueError, TypeError):
        return None


def _declined_offer(text):
    """True only for a clear 'no' to the apply offer. Anything else (yes, or substantive
    detail) proceeds into the flow, where leave_flow still handles a real topic change."""
    t = (text or "").strip().lower()
    return t.startswith("no") or "no thanks" in t or "not now" in t or "maybe later" in t


async def try_apply_answer(request_body):
    """Drive the instant-permit apply flow. Returns None to fall through to normal routing,
    otherwise {"reply": str, "in_flow": bool}. We skip the classifier when already mid-flow."""
    if not PERMIT_APPLY_ENABLED:
        return None
    raw = request_body.get("messages", [])
    if _flow_family(_last_bot_flow(raw)) not in (None, "apply"):
        return None                      # a PRR/inspection flow owns this turn
    # The apply agent re-reads the collected fields from the history each turn, so it needs
    # the WHOLE application, not just the last few turns (else early fields like name/email
    # scroll out of the window and it re-asks). ~30 messages covers a full field-by-field
    # application with slack; the classifier keeps the small window (it only needs recent context).
    history = _recent_history(raw, turns=30)  # user/assistant only; tag tool-message excluded
    user_query = _latest_user_query(raw)
    if not user_query:
        return None
    try:
        client = await init_openai_client()

        async def run_agent():
            result = await apply_agent.answer_apply_query(
                history, client, app_settings.azure_openai.model)
            if result.get("left"):    # user changed topic -> fall through, drop the tag
                return None
            w = result.get("widget")
            # after submit the agent emits a 'pay' widget -> switch to the pay state so the
            # NEXT turn is handled as a payment, not more collection.
            flow = APPLY_PAY if (isinstance(w, dict) and w.get("type") == "pay") else APPLY_FLOW
            return {"reply": result["reply"], "in_flow": result["in_flow"], "widget": w, "flow": flow}

        # 1) mid-flow -> run the agent, classifier skipped
        if _apply_flow_active(raw):
            logging.info("[APPLY AGENT] continuing flow (classifier skipped)")
            return await run_agent()

        # 2) payment step: the pay card was showing and the user tapped Pay (mock). Show the
        # result card + a note that the receipt/permit were emailed to the address they gave.
        if _apply_pay_pending(raw):
            w = _last_bot_widget(raw) or {}
            pn, email = w.get("permitNumber"), w.get("email")
            note = f" A receipt and your permit PDF have been emailed to {email}." if email else ""
            logging.info("[APPLY AGENT] payment received for %s", pn)
            return {"reply": f"Payment received.{note} You can download your permit below.",
                    "in_flow": False, "flow": APPLY_FLOW,
                    "widget": {"type": "result", "permitNumber": pn}}

        # 3) the user is answering the "want help applying?" offer
        if _apply_offer_pending(raw):
            if _declined_offer(user_query):
                logging.info("[APPLY AGENT] offer declined")
                return {"reply": "No problem. If you change your mind or need anything else, just ask.",
                        "in_flow": False, "widget": None, "flow": APPLY_FLOW}
            logging.info("[APPLY AGENT] offer accepted -> entering flow")
            return await run_agent()

        # 4) fresh message -> classify; on instant-permit intent, OFFER first (no RAG, no collection yet)
        if await _domain_for(request_body, client) != "instant-permit":
            return None
        logging.info("[APPLY AGENT] offering apply: %s", user_query)
        offer = ("That requires a permit, and it's one you can file instantly right here. "
                 "Want me to help you apply now?")
        return {"reply": offer, "in_flow": True, "flow": APPLY_OFFER,
                "widget": {"type": "chips", "options": ["Yes, help me apply", "No thanks"]}}
    except Exception:
        logging.exception("apply agent failed; falling back to RAG")
        return None


def _apply_messages(reply, in_flow, widget=None, flow=APPLY_FLOW):
    """Build choices[0].messages: a tag (tool) message first when in_flow, then the reply.
    The tool message is how the flow tag round-trips (see _last_bot_flow), and it also
    carries the interactive `widget` payload for the frontend to render. `flow` is the tag
    value: APPLY_FLOW mid-application, or APPLY_OFFER while the yes/no offer is showing."""
    msgs = []
    if in_flow or widget:
        tag = {}
        if in_flow:               # tag marks the state so the classifier is skipped next turn
            tag["flow"] = flow
        if widget:                # emit the widget even on the final (submitted) turn -> result card
            tag["widget"] = widget
        msgs.append({"role": "tool", "content": json.dumps(tag)})
    msgs.append({"role": "assistant", "content": reply})
    return msgs


def apply_non_streaming_response(reply, in_flow, history_metadata, widget=None, flow=APPLY_FLOW):
    obj = _permit_message_obj()
    obj["id"] = "apply-agent"
    obj["choices"][0]["messages"] = _apply_messages(reply, in_flow, widget, flow)
    obj["history_metadata"] = history_metadata
    obj["apim-request-id"] = "apply-agent"
    return obj


def apply_stream_response(reply, in_flow, history_metadata, widget=None, flow=APPLY_FLOW):
    async def generate():
        obj = _permit_message_obj()
        obj["id"] = "apply-agent"
        obj["object"] = "extensions.chat.completion.chunk"
        obj["choices"][0]["messages"] = _apply_messages(reply, in_flow, widget, flow)
        obj["history_metadata"] = history_metadata
        obj["apim-request-id"] = "apply-agent"
        yield obj
    return generate()


# --- Mock flows: public-record request (offer -> form window) + inspection (conversational) --
def detect_prr_query(query):
    """Explicit public-records intent only, so normal questions aren't hijacked."""
    if not query or not isinstance(query, str):
        return False
    q = query.lower()
    if "prr" in re.split(r"\W+", q):                 # standalone 'prr' token
        return True
    return "public record" in q or "records request" in q


_INSPECTION_VERBS = ("schedule", "reschedule", "book", "move", "set up", "arrange")


def detect_inspection_query(query):
    """Require an inspection noun AND an action verb, so 'what inspections do I need' is ignored."""
    if not query or not isinstance(query, str):
        return False
    q = query.lower()
    if "inspection" not in q and "inspector" not in q:
        return False
    return any(v in q for v in _INSPECTION_VERBS)


async def try_prr_answer(request_body):
    """Public-record request flow: keyword -> offer -> single form window -> review -> mock ref."""
    if not MOCK_FLOWS_ENABLED:
        return None
    raw = request_body.get("messages", [])
    if _flow_family(_last_bot_flow(raw)) not in (None, "prr"):
        return None
    history = _recent_history(raw, turns=30)
    user_query = _latest_user_query(raw)
    if not user_query:
        return None
    try:
        client = await init_openai_client()

        async def run_agent():
            result = await prr_agent.answer_prr_query(history, client, app_settings.azure_openai.model)
            if result.get("left"):
                return None
            return {"reply": result["reply"], "in_flow": result["in_flow"],
                    "widget": result.get("widget"), "flow": PRR_FLOW}

        if _last_bot_flow(raw) == PRR_FLOW:              # mid-flow: run the agent
            return await run_agent()
        if _last_bot_flow(raw) == PRR_OFFER:             # answering the offer
            if _declined_offer(user_query):
                return {"reply": "No problem. If you change your mind or need anything else, just ask.",
                        "in_flow": False, "widget": None, "flow": PRR_FLOW}
            return await run_agent()
        # fresh: keyword OR the classifier must say public-record, then offer before the form
        if not detect_prr_query(user_query) and await _domain_for(request_body, client) != "public-record":
            return None
        logging.info("[PRR AGENT] offering: %s", user_query)
        offer = ("I can help you submit a public records request right here. "
                 "Want me to start one for you?")
        return {"reply": offer, "in_flow": True, "flow": PRR_OFFER,
                "widget": {"type": "chips", "options": ["Yes, help me", "No thanks"]}}
    except Exception:
        logging.exception("prr agent failed; falling back to RAG")
        return None


async def try_inspection_answer(request_body):
    """Inspection flow: keyword -> conversational schedule/reschedule -> mock confirmation."""
    if not MOCK_FLOWS_ENABLED:
        return None
    raw = request_body.get("messages", [])
    if _flow_family(_last_bot_flow(raw)) not in (None, "inspection"):
        return None
    history = _recent_history(raw, turns=30)
    user_query = _latest_user_query(raw)
    if not user_query:
        return None
    try:
        client = await init_openai_client()
        if _last_bot_flow(raw) != INSPECTION_FLOW:       # fresh: keyword OR classifier must say inspection
            if not detect_inspection_query(user_query) and await _domain_for(request_body, client) != "inspection":
                return None
        logging.info("[INSPECTION AGENT] handling: %s", user_query)
        result = await inspection_agent.answer_inspection_query(
            history, client, app_settings.azure_openai.model)
        if result.get("left"):
            return None
        return {"reply": result["reply"], "in_flow": result["in_flow"],
                "widget": result.get("widget"), "flow": INSPECTION_FLOW}
    except Exception:
        logging.exception("inspection agent failed; falling back to RAG")
        return None


# Offer-the-apply-chip: when a normal RAG answer is about one of the five instant-permit
# jobs, we staple a "start the application" chip under it. This is NOT a routing decision.
# The question still goes down the normal RAG path; the chip only posts an apply message
# (entering the existing flow) if the user taps it. Deterministic keyword match, no LLM call.
# Job strings must equal apply_agent.WORK_TYPES keys so set_fields resolves the work type.
# Water heater is listed first so "heat pump water heater" resolves to the heater, not HVAC.
_JOB_KEYWORDS = [
    ("Water Heater Replacement", ("water heater", "hot water heater", "water-heater")),
    ("HVAC", ("hvac", "furnace", "air conditioning", "air conditioner", "ac unit", "heat pump", "central air")),
    ("Panel Upgrade", ("panel upgrade", "electrical panel", "service panel", "breaker panel", "sub panel", "subpanel", "main panel")),
    ("House Rewire", ("rewire", "re-wire", "house wiring", "rewiring")),
    ("House Repipe", ("repipe", "re-pipe", "replumb", "re-plumb", "house piping")),
]


def detect_instant_permit_job(query):
    """Return the WORK_TYPES display name if the query is about one of the five jobs, else None."""
    if not query or not isinstance(query, str):
        return None
    q = query.lower()
    for job, kws in _JOB_KEYWORDS:
        if any(k in q for k in kws):
            return job
    return None


def _merge_apply_chip(formatted, job):
    """Merge an 'Apply for a <job> permit' chip into the citations tool message of a RAG
    response so the frontend renders it under the answer (same tool-tag channel the apply
    flow uses). No-op when this object/chunk carries no tool message (e.g. abstentions)."""
    widget = {"type": "chips", "options": [f"Apply for a {job} permit"]}
    for m in formatted.get("choices", [{}])[0].get("messages", []):
        if m.get("role") == "tool" and isinstance(m.get("content"), str):
            try:
                data = json.loads(m["content"])
            except (ValueError, TypeError):
                data = {}
            if isinstance(data, dict):
                data["widget"] = widget
                m["content"] = json.dumps(data)
    return formatted


async def send_chat_request(request_body, request_headers):
    filtered_messages = []
    messages = request_body.get("messages", [])
    for message in messages:
        if message.get("role") != 'tool':
            filtered_messages.append(message)

    request_body['messages'] = filtered_messages
    model_args = prepare_model_args(request_body, request_headers)

    try:
        azure_openai_client = await init_openai_client()

        # Route this question to the right scoped index (website stays default). Reuse the
        # single cached classification; only classifies here if nothing did so earlier.
        if INDEX_ROUTING_ENABLED and app_settings.datasource and model_args.get("extra_body"):
            domain = await _domain_for(request_body, azure_openai_client)
            routed_index = index_for_domain(domain)
            if routed_index:
                model_args["extra_body"]["data_sources"][0]["parameters"]["index_name"] = routed_index
                logging.info(f"[ROUTED INDEX] {routed_index}")

        raw_response = await azure_openai_client.chat.completions.with_raw_response.create(**model_args)
        response = raw_response.parse()
        apim_request_id = raw_response.headers.get("apim-request-id")

        if not app_settings.azure_openai.stream:
            message = response.choices[0].message
            logging.info(f"[OPENAI RESPONSE] {message.content}")
            context = getattr(message, "context", None)
            if context:
                if isinstance(context, dict) and context.get("intent"):
                    logging.info(f"[REFORMULATED QUERY] {context['intent']}")
                logging.info(f"[RETRIEVED CHUNKS] {json.dumps(context, indent=2)}")
    except Exception as e:
        logging.exception("Exception in send_chat_request")
        raise e

    return response, apim_request_id


async def complete_chat_request(request_body, request_headers):
    if app_settings.base_settings.use_promptflow:
        response = await promptflow_request(request_body)
        history_metadata = request_body.get("history_metadata", {})
        return format_pf_non_streaming_response(
            response,
            history_metadata,
            app_settings.promptflow.response_field_name,
            app_settings.promptflow.citations_field_name
        )
    else:
        history_metadata = request_body.get("history_metadata", {})
        # Keyword-triggered mock flows first: their detection is specific, and 'apply for a
        # prr' would otherwise pattern-match the instant-permit classifier and get hijacked.
        prr_answer = await try_prr_answer(request_body)
        if prr_answer is not None:
            return apply_non_streaming_response(prr_answer["reply"], prr_answer["in_flow"],
                                                history_metadata, prr_answer.get("widget"),
                                                prr_answer.get("flow", PRR_FLOW))
        inspection_answer = await try_inspection_answer(request_body)
        if inspection_answer is not None:
            return apply_non_streaming_response(inspection_answer["reply"], inspection_answer["in_flow"],
                                                history_metadata, inspection_answer.get("widget"),
                                                inspection_answer.get("flow", INSPECTION_FLOW))
        apply_answer = await try_apply_answer(request_body)
        if apply_answer is not None:
            return apply_non_streaming_response(apply_answer["reply"], apply_answer["in_flow"],
                                                history_metadata, apply_answer.get("widget"),
                                                apply_answer.get("flow", APPLY_FLOW))
        permit_answer = await try_permit_answer(request_body)
        if permit_answer is not None:
            return permit_non_streaming_response(permit_answer, history_metadata)
        meetings_answer = await try_meetings_answer(request_body)
        if meetings_answer is not None:
            return permit_non_streaming_response(meetings_answer, history_metadata)
        # Detect the job BEFORE send_chat_request (it filters/reassigns request_body messages).
        job = detect_instant_permit_job(_latest_user_query(request_body.get("messages", []))) \
            if PERMIT_APPLY_ENABLED else None
        response, apim_request_id = await send_chat_request(request_body, request_headers)
        rag = format_non_streaming_response(response, history_metadata, apim_request_id)
        if job:
            _merge_apply_chip(rag, job)
        return rag


async def stream_chat_request(request_body, request_headers):
    history_metadata = request_body.get("history_metadata", {})
    # Keyword-triggered mock flows first (see complete_chat_request note).
    prr_answer = await try_prr_answer(request_body)
    if prr_answer is not None:
        return apply_stream_response(prr_answer["reply"], prr_answer["in_flow"],
                                     history_metadata, prr_answer.get("widget"),
                                     prr_answer.get("flow", PRR_FLOW))
    inspection_answer = await try_inspection_answer(request_body)
    if inspection_answer is not None:
        return apply_stream_response(inspection_answer["reply"], inspection_answer["in_flow"],
                                     history_metadata, inspection_answer.get("widget"),
                                     inspection_answer.get("flow", INSPECTION_FLOW))
    apply_answer = await try_apply_answer(request_body)
    if apply_answer is not None:
        return apply_stream_response(apply_answer["reply"], apply_answer["in_flow"],
                                     history_metadata, apply_answer.get("widget"),
                                     apply_answer.get("flow", APPLY_FLOW))
    permit_answer = await try_permit_answer(request_body)
    if permit_answer is not None:
        return permit_stream_response(permit_answer, history_metadata)
    meetings_answer = await try_meetings_answer(request_body)
    if meetings_answer is not None:
        return permit_stream_response(meetings_answer, history_metadata)
    # Detect the job BEFORE send_chat_request (it filters/reassigns request_body messages).
    job = detect_instant_permit_job(_latest_user_query(request_body.get("messages", []))) \
        if PERMIT_APPLY_ENABLED else None
    response, apim_request_id = await send_chat_request(request_body, request_headers)

    async def generate():
        context_logged = False
        full_response = []
        async for completionChunk in response:
            if len(completionChunk.choices) > 0:
                delta = completionChunk.choices[0].delta
                if not context_logged:
                    context = getattr(delta, "context", None)
                    if context:
                        if isinstance(context, dict) and context.get("intent"):
                            logging.info(f"[REFORMULATED QUERY] {context['intent']}")
                        logging.info(f"[RETRIEVED CHUNKS] {json.dumps(context, indent=2)}")
                        context_logged = True
                if getattr(delta, "content", None):
                    full_response.append(delta.content)
            formatted = format_stream_response(completionChunk, history_metadata, apim_request_id)
            if job:                       # merge the chip into the citations tool chunk
                _merge_apply_chip(formatted, job)
            yield formatted
        if full_response:
            logging.info(f"[OPENAI RESPONSE] {''.join(full_response)}")

    return generate()


async def conversation_internal(request_body, request_headers):
    try:
        if app_settings.azure_openai.stream and not app_settings.base_settings.use_promptflow:
            result = await stream_chat_request(request_body, request_headers)
            response = await make_response(format_as_ndjson(result))
            response.timeout = None
            response.mimetype = "application/json-lines"
            return response
        else:
            result = await complete_chat_request(request_body, request_headers)
            return jsonify(result)

    except Exception as ex:
        logging.exception(ex)
        if hasattr(ex, "status_code"):
            return jsonify({"error": str(ex)}), ex.status_code
        else:
            return jsonify({"error": str(ex)}), 500


@bp.route("/conversation", methods=["POST"])
async def conversation():
    if not request.is_json:
        return jsonify({"error": "request must be json"}), 415
    request_json = await request.get_json()

    return await conversation_internal(request_json, request.headers)


@bp.route("/frontend_settings", methods=["GET"])
def get_frontend_settings():
    try:
        return jsonify(frontend_settings), 200
    except Exception as e:
        logging.exception("Exception in /frontend_settings")
        return jsonify({"error": str(e)}), 500


@bp.route("/apply/address-search", methods=["GET"])
async def apply_address_search():
    """Type-ahead search for the instant-permit address widget. Backed by the once-cached
    getAllAddresses list (prefix-first), so it never hits the permit API per keystroke.
    Returns [{address, lsoId, apn}] (empty for queries under 2 chars)."""
    if not PERMIT_APPLY_ENABLED or apply_agent is None:
        return jsonify([]), 200
    try:
        from backend.permit_agent import apply_client
        results = await apply_client.search_addresses(request.args.get("q", ""), limit=8)
        return jsonify(results), 200
    except Exception:
        logging.exception("apply address search failed")
        return jsonify([]), 200


@bp.route("/apply/permit-pdf", methods=["GET"])
async def apply_permit_pdf():
    """Download the permit PDF (proxies the permit API's getPermitReport) for the result widget."""
    if not PERMIT_APPLY_ENABLED or apply_agent is None:
        return "", 404
    from quart import Response
    from backend.permit_agent import apply_client
    permit = request.args.get("permit", "")
    try:
        pdf = await apply_client.permit_report(permit)
    except Exception:
        logging.exception("permit pdf fetch failed")
        return jsonify({"error": "report unavailable"}), 502
    return Response(pdf, mimetype="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="permit-{permit}.pdf"'})


## Conversation History API ##
@bp.route("/history/generate", methods=["POST"])
async def add_conversation():
    await cosmos_db_ready.wait()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    ## check request for conversation_id
    request_json = await request.get_json()
    conversation_id = request_json.get("conversation_id", None)

    try:
        # make sure cosmos is configured
        if not current_app.cosmos_conversation_client:
            raise Exception("CosmosDB is not configured or not working")

        # check for the conversation_id, if the conversation is not set, we will create a new one
        history_metadata = {}
        if not conversation_id:
            title = await generate_title(request_json["messages"])
            conversation_dict = await current_app.cosmos_conversation_client.create_conversation(
                user_id=user_id, title=title
            )
            conversation_id = conversation_dict["id"]
            history_metadata["title"] = title
            history_metadata["date"] = conversation_dict["createdAt"]

        ## Format the incoming message object in the "chat/completions" messages format
        ## then write it to the conversation history in cosmos
        messages = request_json["messages"]
        if len(messages) > 0 and messages[-1]["role"] == "user":
            createdMessageValue = await current_app.cosmos_conversation_client.create_message(
                uuid=str(uuid.uuid4()),
                conversation_id=conversation_id,
                user_id=user_id,
                input_message=messages[-1],
            )
            if createdMessageValue == "Conversation not found":
                raise Exception(
                    "Conversation not found for the given conversation ID: "
                    + conversation_id
                    + "."
                )
        else:
            raise Exception("No user message found")

        # Submit request to Chat Completions for response
        request_body = await request.get_json()
        history_metadata["conversation_id"] = conversation_id
        request_body["history_metadata"] = history_metadata
        return await conversation_internal(request_body, request.headers)

    except Exception as e:
        logging.exception("Exception in /history/generate")
        return jsonify({"error": str(e)}), 500


@bp.route("/history/update", methods=["POST"])
async def update_conversation():
    await cosmos_db_ready.wait()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    ## check request for conversation_id
    request_json = await request.get_json()
    conversation_id = request_json.get("conversation_id", None)

    try:
        # make sure cosmos is configured
        if not current_app.cosmos_conversation_client:
            raise Exception("CosmosDB is not configured or not working")

        # check for the conversation_id, if the conversation is not set, we will create a new one
        if not conversation_id:
            raise Exception("No conversation_id found")

        ## Format the incoming message object in the "chat/completions" messages format
        ## then write it to the conversation history in cosmos
        messages = request_json["messages"]
        if len(messages) > 0 and messages[-1]["role"] == "assistant":
            if len(messages) > 1 and messages[-2].get("role", None) == "tool":
                # write the tool message first
                await current_app.cosmos_conversation_client.create_message(
                    uuid=str(uuid.uuid4()),
                    conversation_id=conversation_id,
                    user_id=user_id,
                    input_message=messages[-2],
                )
            # write the assistant message
            await current_app.cosmos_conversation_client.create_message(
                uuid=messages[-1]["id"],
                conversation_id=conversation_id,
                user_id=user_id,
                input_message=messages[-1],
            )
        else:
            raise Exception("No bot messages found")

        # Submit request to Chat Completions for response
        response = {"success": True}
        return jsonify(response), 200

    except Exception as e:
        logging.exception("Exception in /history/update")
        return jsonify({"error": str(e)}), 500


@bp.route("/history/message_feedback", methods=["POST"])
async def update_message():
    await cosmos_db_ready.wait()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    ## check request for message_id
    request_json = await request.get_json()
    message_id = request_json.get("message_id", None)
    message_feedback = request_json.get("message_feedback", None)    
    other_text = request_json.get("other_text", None)
    
    try:
        if not message_id:
            return jsonify({"error": "message_id is required"}), 400

        if not message_feedback:
            return jsonify({"error": "message_feedback is required"}), 400

        ## update the message in cosmos
        updated_message = await current_app.cosmos_conversation_client.update_message_feedback(
            user_id, message_id, message_feedback, other_text
        )
        if updated_message:
            return (
                jsonify(
                    {
                        "message": f"Successfully updated message with feedback {message_feedback}",
                        "message_id": message_id,
                    }
                ),
                200,
            )
        else:
            return (
                jsonify(
                    {
                        "error": f"Unable to update message {message_id}. It either does not exist or the user does not have access to it."
                    }
                ),
                404,
            )

    except Exception as e:
        logging.exception("Exception in /history/message_feedback")
        return jsonify({"error": str(e)}), 500


@bp.route("/history/delete", methods=["DELETE"])
async def delete_conversation():
    await cosmos_db_ready.wait()
    ## get the user id from the request headers
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    ## check request for conversation_id
    request_json = await request.get_json()
    conversation_id = request_json.get("conversation_id", None)

    try:
        if not conversation_id:
            return jsonify({"error": "conversation_id is required"}), 400

        ## make sure cosmos is configured
        if not current_app.cosmos_conversation_client:
            raise Exception("CosmosDB is not configured or not working")

        ## delete the conversation messages from cosmos first
        deleted_messages = await current_app.cosmos_conversation_client.delete_messages(
            conversation_id, user_id
        )

        ## Now delete the conversation
        deleted_conversation = await current_app.cosmos_conversation_client.delete_conversation(
            user_id, conversation_id
        )

        return (
            jsonify(
                {
                    "message": "Successfully deleted conversation and messages",
                    "conversation_id": conversation_id,
                }
            ),
            200,
        )
    except Exception as e:
        logging.exception("Exception in /history/delete")
        return jsonify({"error": str(e)}), 500


@bp.route("/history/list", methods=["GET"])
async def list_conversations():
    await cosmos_db_ready.wait()
    offset = request.args.get("offset", 0)
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    ## make sure cosmos is configured
    if not current_app.cosmos_conversation_client:
        raise Exception("CosmosDB is not configured or not working")

    ## get the conversations from cosmos
    conversations = await current_app.cosmos_conversation_client.get_conversations(
        user_id, offset=offset, limit=25
    )
    if not isinstance(conversations, list):
        return jsonify({"error": f"No conversations for {user_id} were found"}), 404

    ## return the conversation ids

    return jsonify(conversations), 200


@bp.route("/history/read", methods=["POST"])
async def get_conversation():
    await cosmos_db_ready.wait()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    ## check request for conversation_id
    request_json = await request.get_json()
    conversation_id = request_json.get("conversation_id", None)

    if not conversation_id:
        return jsonify({"error": "conversation_id is required"}), 400

    ## make sure cosmos is configured
    if not current_app.cosmos_conversation_client:
        raise Exception("CosmosDB is not configured or not working")

    ## get the conversation object and the related messages from cosmos
    conversation = await current_app.cosmos_conversation_client.get_conversation(
        user_id, conversation_id
    )
    ## return the conversation id and the messages in the bot frontend format
    if not conversation:
        return (
            jsonify(
                {
                    "error": f"Conversation {conversation_id} was not found. It either does not exist or the logged in user does not have access to it."
                }
            ),
            404,
        )

    # get the messages for the conversation from cosmos
    conversation_messages = await current_app.cosmos_conversation_client.get_messages(
        user_id, conversation_id
    )

    ## format the messages in the bot frontend format
    messages = [
        {
            "id": msg["id"],
            "role": msg["role"],
            "content": msg["content"],
            "createdAt": msg["createdAt"],
            "feedback": msg.get("feedback"),
        }
        for msg in conversation_messages
    ]

    return jsonify({"conversation_id": conversation_id, "messages": messages}), 200


@bp.route("/history/rename", methods=["POST"])
async def rename_conversation():
    await cosmos_db_ready.wait()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    ## check request for conversation_id
    request_json = await request.get_json()
    conversation_id = request_json.get("conversation_id", None)

    if not conversation_id:
        return jsonify({"error": "conversation_id is required"}), 400

    ## make sure cosmos is configured
    if not current_app.cosmos_conversation_client:
        raise Exception("CosmosDB is not configured or not working")

    ## get the conversation from cosmos
    conversation = await current_app.cosmos_conversation_client.get_conversation(
        user_id, conversation_id
    )
    if not conversation:
        return (
            jsonify(
                {
                    "error": f"Conversation {conversation_id} was not found. It either does not exist or the logged in user does not have access to it."
                }
            ),
            404,
        )

    ## update the title
    title = request_json.get("title", None)
    if not title:
        return jsonify({"error": "title is required"}), 400
    conversation["title"] = title
    updated_conversation = await current_app.cosmos_conversation_client.upsert_conversation(
        conversation
    )

    return jsonify(updated_conversation), 200


@bp.route("/history/delete_all", methods=["DELETE"])
async def delete_all_conversations():
    await cosmos_db_ready.wait()
    ## get the user id from the request headers
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    # get conversations for user
    try:
        ## make sure cosmos is configured
        if not current_app.cosmos_conversation_client:
            raise Exception("CosmosDB is not configured or not working")

        conversations = await current_app.cosmos_conversation_client.get_conversations(
            user_id, offset=0, limit=None
        )
        if not conversations:
            return jsonify({"error": f"No conversations for {user_id} were found"}), 404

        # delete each conversation
        for conversation in conversations:
            ## delete the conversation messages from cosmos first
            deleted_messages = await current_app.cosmos_conversation_client.delete_messages(
                conversation["id"], user_id
            )

            ## Now delete the conversation
            deleted_conversation = await current_app.cosmos_conversation_client.delete_conversation(
                user_id, conversation["id"]
            )
        return (
            jsonify(
                {
                    "message": f"Successfully deleted conversation and messages for user {user_id}"
                }
            ),
            200,
        )

    except Exception as e:
        logging.exception("Exception in /history/delete_all")
        return jsonify({"error": str(e)}), 500


@bp.route("/history/clear", methods=["POST"])
async def clear_messages():
    await cosmos_db_ready.wait()
    ## get the user id from the request headers
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    ## check request for conversation_id
    request_json = await request.get_json()
    conversation_id = request_json.get("conversation_id", None)

    try:
        if not conversation_id:
            return jsonify({"error": "conversation_id is required"}), 400

        ## make sure cosmos is configured
        if not current_app.cosmos_conversation_client:
            raise Exception("CosmosDB is not configured or not working")

        ## delete the conversation messages from cosmos
        deleted_messages = await current_app.cosmos_conversation_client.delete_messages(
            conversation_id, user_id
        )

        return (
            jsonify(
                {
                    "message": "Successfully deleted messages in conversation",
                    "conversation_id": conversation_id,
                }
            ),
            200,
        )
    except Exception as e:
        logging.exception("Exception in /history/clear_messages")
        return jsonify({"error": str(e)}), 500


@bp.route("/history/ensure", methods=["GET"])
async def ensure_cosmos():
    await cosmos_db_ready.wait()
    if not app_settings.chat_history:
        return jsonify({"error": "CosmosDB is not configured"}), 404

    try:
        success, err = await current_app.cosmos_conversation_client.ensure()
        if not current_app.cosmos_conversation_client or not success:
            if err:
                return jsonify({"error": err}), 422
            return jsonify({"error": "CosmosDB is not configured or not working"}), 500

        return jsonify({"message": "CosmosDB is configured and working"}), 200
    except Exception as e:
        logging.exception("Exception in /history/ensure")
        cosmos_exception = str(e)
        if "Invalid credentials" in cosmos_exception:
            return jsonify({"error": cosmos_exception}), 401
        elif "Invalid CosmosDB database name" in cosmos_exception:
            return (
                jsonify(
                    {
                        "error": f"{cosmos_exception} {app_settings.chat_history.database} for account {app_settings.chat_history.account}"
                    }
                ),
                422,
            )
        elif "Invalid CosmosDB container name" in cosmos_exception:
            return (
                jsonify(
                    {
                        "error": f"{cosmos_exception}: {app_settings.chat_history.conversations_container}"
                    }
                ),
                422,
            )
        else:
            return jsonify({"error": "CosmosDB is not working"}), 500


async def generate_title(conversation_messages) -> str:
    ## make sure the messages are sorted by _ts descending
    title_prompt = "Summarize the conversation so far into a 4-word or less title. Do not use any quotation marks or punctuation. Do not include any other commentary or description."

    messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in conversation_messages
    ]
    messages.append({"role": "user", "content": title_prompt})

    try:
        azure_openai_client = await init_openai_client()
        response = await azure_openai_client.chat.completions.create(
            model=app_settings.azure_openai.model, messages=messages, temperature=1, max_tokens=64
        )

        title = response.choices[0].message.content
        return title
    except Exception as e:
        logging.exception("Exception while generating title", e)
        return messages[-2]["content"]


app = create_app()
