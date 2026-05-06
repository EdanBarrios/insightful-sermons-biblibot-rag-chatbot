import os
import logging
from groq import Groq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- Constants --------------------

_GREETINGS = frozenset(['hi', 'hello', 'hey', 'yo', 'sup', 'howdy', 'greetings'])

_SYSTEM_SERMON = """You are BibliBot, a Biblical counselor and spiritual guide. Use the sermon content provided to give structured, actionable answers.

ANSWER FORMAT (follow exactly — plain text only, no asterisks or markdown):

Quick Answer:
[1-2 sentences directly answering the question with a clear, memorable takeaway]

Your Path Forward:
• [Concrete action or concept] — [weave in a scripture reference or sermon insight naturally]
• [Concrete action or concept] — [weave in a scripture reference or sermon insight naturally]
• [Concrete action or concept] — [weave in a scripture reference or sermon insight naturally]
(3–5 bullets; vary phrasing so they don't all sound the same)

Theological Foundation:
[1-2 sentences connecting this to a broader Biblical theme. If the sermon context names an author, reference them: "As [Author] reminds us in their sermon on [topic], ..."]

RULES:
- Use ONLY content drawn from the sermon context above
- Weave scripture naturally into bullet points — don't just append citations at the end
- If a Bible verse is provided, integrate it into the most fitting bullet point
- Reference sermon authors by name when the context includes one
- Never use vague filler phrases like "is an important aspect of" or "can be challenging"
- Never give advice without a concrete step ("trust God" must come with a how)
- Be pastoral and compassionate — frame guidance gently, never accusatorily
- Keep total response under 300 words
- Do NOT use ** or any markdown formatting — plain text only"""

_SYSTEM_GENERAL = """You are BibliBot, a Biblical counselor and spiritual guide. Our sermon library does not have specific content on this topic, so answer using your general Biblical and theological knowledge.

ANSWER FORMAT (plain text only, no asterisks or markdown):

Quick Answer:
[1-2 sentences directly answering the question]

Your Path Forward:
• [Insight or action] — [weave in a scripture reference naturally]
• [Insight or action] — [weave in a scripture reference naturally]
• [Insight or action] — [weave in a scripture reference naturally]

Note: Our sermon library doesn't have specific content on this topic. You might find related sermons by asking about [suggest 1-2 closely related Biblical themes].

RULES:
- Ground every point in scripture — cite verses naturally, never fabricate sermon titles or authors
- If you genuinely don't know, say so plainly rather than guessing
- Keep total response under 250 words
- Do NOT use ** or any markdown formatting — plain text only"""

# -------------------- Client --------------------

try:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    logger.info("✅ Groq client initialized")
except Exception as e:
    logger.error(f"❌ Groq initialization failed: {e}")
    client = None

# -------------------- Functions --------------------

def is_sermon_question(question: str) -> bool:
    if question.lower().strip() in _GREETINGS:
        return False
    if len(question.split()) < 3 and '?' not in question:
        return False
    return True


def generate_answer(context: str, question: str, has_sermon_content: bool = True, bible_verse_context: str = "") -> str:
    if not client:
        return "I'm having trouble connecting to the AI service. Please try again later."

    if not is_sermon_question(question):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "You are BibliBot, a friendly assistant. Respond warmly in 1-2 sentences and invite them to ask about Biblical topics."
                    },
                    {"role": "user", "content": question}
                ],
                temperature=0.8,
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"❌ Small talk error: {e}")
            return "Hello! I'm BibliBot. Ask me about faith, grace, prayer, or any Biblical topic!"

    if not has_sermon_content or not context.strip():
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": _SYSTEM_GENERAL},
                    {"role": "user", "content": question}
                ],
                temperature=0.4,
                max_tokens=500,
                top_p=0.9
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"❌ General knowledge fallback error: {e}")
            return "I don't have sermons on this topic. Try asking about faith, grace, prayer, love, or hope."

    bible_section = f"\nBIBLE VERSE:\n{bible_verse_context}\n" if bible_verse_context else ""
    prompt = f"""SERMON CONTEXT:
{context}
{bible_section}
QUESTION: {question}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _SYSTEM_SERMON},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=600,
            top_p=0.9
        )
        answer = response.choices[0].message.content.strip()
        logger.info(f"✅ Generated answer: {answer[:100]}...")
        return answer
    except Exception as e:
        logger.error(f"❌ LLM generation error: {e}")
        return "I apologize, but I'm having trouble generating a response right now. Please try again."
