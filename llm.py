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

def generate_answer(context: str, question: str) -> str:
    """
    Generate answer using Groq's API with smart routing.
    
    Args:
        context: Retrieved sermon chunks (may be empty)
        question: User's question
        
    Returns:
        Generated answer
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
                        "content": "You are BibliBot, a friendly assistant that helps people explore Biblical sermons and teachings. When greeted, respond warmly and invite them to ask questions about faith, sermons, or Biblical topics."
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ],
                temperature=0.8,
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"❌ Small talk error: {e}")
            return "Hello! I'm BibliBot. I'm here to help you explore Biblical sermons and teachings. What would you like to know about?"
    
    # Handle sermon questions with no context
    if not context.strip():
        return "I couldn't find any relevant sermon content to answer that specific question. I can help you with topics like faith, grace, prayer, love, hope, and other Biblical teachings. Could you rephrase your question or try a different topic?"
    
    # Handle sermon questions with context (RAG mode)
    prompt = f"""You are BibliBot, a helpful assistant answering questions about Biblical sermons.

Use the following sermon excerpts to answer the question. Be conversational, clear, and faithful to the content.

SERMON EXCERPTS:
{context}

QUESTION:
{question}

ANSWER (be warm and helpful):"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are BibliBot, a knowledgeable and warm assistant helping people understand Biblical sermons. Provide clear, helpful answers based on the sermon content provided. Be conversational but stay grounded in the sermons."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=400,
            top_p=0.9
        )
        
        answer = response.choices[0].message.content.strip()
        logger.info(f"✅ Generated answer: {answer[:100]}...")
        return answer
        
    except Exception as e:
        logger.error(f"❌ LLM generation error: {e}")
        return "I apologize, but I'm having trouble generating a response right now. Please try again in a moment."