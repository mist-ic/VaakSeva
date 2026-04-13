"""
Prompt templates for VaakSeva.

All prompts are defined here. No prompt strings in other files.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# System prompt (VaakSeva persona)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """आप VaakSeva हैं, एक सहायक हिंदी AI जो भारतीय सरकारी योजनाओं के बारे में सवालों के जवाब देता है।

नियम:
1. केवल नीचे दिए गए संदर्भ दस्तावेज़ों के आधार पर उत्तर दें
2. यदि उत्तर संदर्भ में नहीं है, तो कहें "इस बारे में मेरे पास जानकारी नहीं है"
3. हमेशा हिंदी में उत्तर दें
4. योजना का नाम और लाभ राशि बताते समय सटीक रहें
5. आवेदन करने की प्रक्रिया भी बताएं यदि संदर्भ में उपलब्ध हो
6. व्यक्तिगत जानकारी (नाम, आधार नंबर) कभी न मांगें
7. संक्षिप्त और स्पष्ट रहें - 3-5 वाक्यों में उत्तर दें"""

# ---------------------------------------------------------------------------
# RAG prompt template
# ---------------------------------------------------------------------------

RAG_PROMPT_TEMPLATE = """\
संदर्भ दस्तावेज़:
{context}

---

उपयोगकर्ता की जानकारी:
{user_profile_str}

पिछली बातचीत:
{history_str}

उपयोगकर्ता का प्रश्न: {query}

उपरोक्त संदर्भ दस्तावेज़ों के आधार पर हिंदी में जवाब दें:"""

# ---------------------------------------------------------------------------
# Profile extraction prompt
# ---------------------------------------------------------------------------

PROFILE_EXTRACTION_PROMPT_TEMPLATE = """\
निम्नलिखित हिंदी संदेश से व्यक्तिगत जानकारी निकालें।
केवल वही जानकारी शामिल करें जो स्पष्ट रूप से उल्लेखित हो।
यदि कोई जानकारी नहीं है तो खाली JSON लौटाएं: {{}}

संभावित फ़ील्ड:
- age (उम्र, number)
- gender (लिंग: male/female/other)
- state (राज्य, string)
- district (जिला, string)
- occupation (व्यवसाय: farmer/laborer/student/unemployed/other)
- income (वार्षिक आय, number in rupees)
- category (SC/ST/OBC/General)
- education (शिक्षा: none/primary/secondary/graduate/postgraduate)
- has_aadhaar (आधार है?: true/false)
- is_bpl (बीपीएल में है?: true/false)

संदेश: "{message}"

JSON उत्तर:"""

# ---------------------------------------------------------------------------
# Eligibility query prompt
# ---------------------------------------------------------------------------

ELIGIBILITY_PROMPT_TEMPLATE = """\
उपयोगकर्ता की जानकारी के आधार पर बताएं कि वे किन सरकारी योजनाओं के लिए पात्र हैं।

उपयोगकर्ता की जानकारी:
{user_profile_str}

पात्र योजनाएं:
{eligible_schemes_str}

हिंदी में संक्षिप्त उत्तर दें जिसमें हर योजना का नाम, मुख्य लाभ, और आवेदन कैसे करें यह शामिल हो:"""


# ---------------------------------------------------------------------------
# Builder functions
# ---------------------------------------------------------------------------


def build_rag_prompt(
    query: str,
    context: str,
    user_profile: dict | None = None,
    conversation_history: list | None = None,
) -> str:
    """Build the full RAG prompt with context, user profile, and history."""
    user_profile = user_profile or {}
    conversation_history = conversation_history or []

    # Format user profile
    if user_profile:
        profile_lines = []
        field_labels = {
            "age": "उम्र",
            "gender": "लिंग",
            "state": "राज्य",
            "district": "जिला",
            "occupation": "व्यवसाय",
            "income": "वार्षिक आय",
            "category": "श्रेणी",
            "education": "शिक्षा",
        }
        for field, label in field_labels.items():
            value = user_profile.get(field)
            if value is not None:
                if field == "income":
                    profile_lines.append(f"{label}: ₹{value:,}")
                else:
                    profile_lines.append(f"{label}: {value}")
        user_profile_str = "\n".join(profile_lines) if profile_lines else "जानकारी उपलब्ध नहीं"
    else:
        user_profile_str = "जानकारी उपलब्ध नहीं"

    # Format last N conversation turns
    if conversation_history:
        recent = conversation_history[-4:]  # last 2 exchanges
        history_lines = []
        for turn in recent:
            role_label = "उपयोगकर्ता" if turn.get("role") == "user" else "VaakSeva"
            history_lines.append(f"{role_label}: {turn.get('content', '')[:200]}")
        history_str = "\n".join(history_lines)
    else:
        history_str = "कोई पिछली बातचीत नहीं"

    return RAG_PROMPT_TEMPLATE.format(
        context=context,
        user_profile_str=user_profile_str,
        history_str=history_str,
        query=query,
    )


def build_profile_extraction_prompt(message: str) -> str:
    """Build prompt for extracting user profile from a Hindi message."""
    return PROFILE_EXTRACTION_PROMPT_TEMPLATE.format(message=message)


def build_eligibility_prompt(user_profile: dict, eligible_schemes: list[dict]) -> str:
    """Build prompt for explaining eligibility results to user."""
    user_profile_str = "\n".join(
        f"{k}: {v}" for k, v in user_profile.items() if v is not None
    ) or "जानकारी उपलब्ध नहीं"

    schemes_parts = []
    for scheme in eligible_schemes[:5]:  # top 5
        schemes_parts.append(
            f"- {scheme.get('scheme_name', '')} (पात्रता स्कोर: {scheme.get('score', 0):.0%})\n"
            f"  लाभ: {scheme.get('benefits_summary_hi', '')}"
        )
    eligible_schemes_str = "\n".join(schemes_parts) or "कोई योजना नहीं मिली"

    return ELIGIBILITY_PROMPT_TEMPLATE.format(
        user_profile_str=user_profile_str,
        eligible_schemes_str=eligible_schemes_str,
    )
