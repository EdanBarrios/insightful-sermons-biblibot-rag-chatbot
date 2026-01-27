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

def generate_answer(context: str, question: str, has_sermon_content: bool = True) -> str:
    """
    Generate answer - with or without sermon content.
    
    Args:
        context: Retrieved sermon chunks (may be empty)
        question: User's question
        has_sermon_content: Whether we found relevant sermons
        
    Returns:
        Answer that acknowledges sermon availability
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
                        "content": "You are BibliBot, a friendly assistant. Respond warmly in 1-2 sentences and invite them to ask about Biblical topics."
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
    
    # Handle questions WITHOUT sermon content - REFUSE to answer
    if not has_sermon_content or not context.strip():
        return "I don't have any sermons that cover this topic. Try asking about faith, grace, prayer, love, hope, or other Biblical themes from our sermon library."
    
    # Handle questions WITH sermon content - use sermons!
    prompt = f"""You are BibliBot. Answer based on these sermon excerpts.

SERMON EXCERPTS:
{context}

QUESTION:
{question}

Provide a brief answer (2-3 sentences) using ONLY the information in the sermon excerpts above. Be conversational but stay strictly within the sermon content."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are BibliBot. Answer in 2-3 sentences using ONLY the sermon content provided. Be conversational and helpful."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=150,
            top_p=0.9
        )
        
        answer = response.choices[0].message.content.strip()
        logger.info(f"✅ Generated answer (with sermons): {answer[:100]}...")
        return answer
        
    except Exception as e:
        logger.error(f"❌ LLM generation error: {e}")
        return "I apologize, but I'm having trouble generating a response right now. Please try again."