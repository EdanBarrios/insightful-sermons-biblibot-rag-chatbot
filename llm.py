import os
import logging
from groq import Groq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Groq client
try:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    logger.info("✅ Groq client initialized")
except Exception as e:
    logger.error(f"❌ Groq initialization failed: {e}")
    client = None

def is_sermon_question(question: str) -> bool:
    """Quick check if question is sermon/Bible related"""
    question_lower = question.lower()
    
    # Greetings and small talk
    greetings = ['hi', 'hello', 'hey', 'yo', 'sup', 'howdy', 'greetings']
    if any(question_lower.strip() == g for g in greetings):
        return False
    
    # Very short non-questions
    if len(question.split()) < 3 and '?' not in question:
        return False
    
    # Otherwise assume it's a real question
    return True

def generate_answer(context: str, question: str, sources: list[dict] = None) -> str:
    """
    Generate SHORT answer with sermon references.
    
    Args:
        context: Retrieved sermon chunks
        question: User's question
        sources: List of dicts with 'title', 'url', 'category'
        
    Returns:
        Brief answer with sermon links
    """
    
    if not client:
        return "I'm having trouble connecting to the AI service. Please try again later."
    
    # Handle greetings and small talk
    if not is_sermon_question(question):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "You are BibliBot, a friendly assistant that helps people explore Biblical sermons. When greeted, respond warmly in 1-2 sentences and invite them to ask about faith topics."
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ],
                temperature=0.8,
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"❌ Small talk error: {e}")
            return "Hello! I'm BibliBot. Ask me about faith, grace, prayer, or any Biblical topic!"
    
    # Handle sermon questions with no context
    if not context.strip():
        return "I couldn't find any sermons about that topic. Try asking about faith, grace, prayer, love, hope, or other Biblical themes."
    
    # Handle sermon questions with context - STRICT SERMON-ONLY MODE
    prompt = f"""You are BibliBot. Answer ONLY using information from these sermon excerpts. Do NOT add any information from your general knowledge.

SERMON EXCERPTS:
{context}

QUESTION:
{question}

CRITICAL RULES:
- ONLY use information explicitly stated in the sermon excerpts above
- If the excerpts don't contain enough information to answer, say "I don't have enough information in our sermons about that specific aspect"
- Do NOT add general Biblical knowledge, personal interpretations, or information not in the excerpts
- Keep answer to 2-3 sentences maximum
- Be conversational but stay strictly within the sermon content
- Do NOT mention sermon titles or URLs - those will be added separately"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are BibliBot. You ONLY answer based on the sermon excerpts provided. Never use your general knowledge or add information not explicitly in the excerpts. If the excerpts don't answer the question, say so. Keep responses to 2-3 sentences."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,  # Lowered from 0.7 for more factual responses
            max_tokens=150,
            top_p=0.9
        )
        
        answer = response.choices[0].message.content.strip()
        logger.info(f"✅ Generated answer: {answer[:100]}...")
        return answer
        
    except Exception as e:
        logger.error(f"❌ LLM generation error: {e}")
        return "I apologize, but I'm having trouble generating a response right now. Please try again."