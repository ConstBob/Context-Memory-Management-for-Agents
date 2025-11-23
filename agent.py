from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import os, time, random
import re, ast, math
from tavily import TavilyClient
from google import genai
from google.genai import types
import datetime
import requests
import json

class RateLimiter:
    def __init__(self, min_interval_sec=0.2):
        self.min_interval = float(min_interval_sec)
        self._last = 0.0
    def acquire(self):
        now = time.time()
        wait = self.min_interval - (now - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()

def _should_retry(e_msg: str) -> bool:
    s = (e_msg or "").lower()
    return ("resource_exhausted" in s) or ("rate" in s) or ("quota" in s) or ("429" in s) or ("exceeded" in s)

def _retry_with_backoff(fn, logger, max_retries=4, base=0.5, jitter=0.2):
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            logger.log(f"[retry] attempt={attempt} error={msg}")
            if attempt >= max_retries or not _should_retry(msg):
                raise
            sleep = base * (2 ** attempt) + random.uniform(0, jitter)
            time.sleep(sleep)

class OpenRouterClient:
    """OpenRouter API client for multiple LLM providers"""
    def __init__(self, api_key: str, model_name: str, max_new_tokens: int = 512, limiter: RateLimiter | None = None, logger=None):
        self.api_key = api_key
        self.model = model_name
        self.max_new_tokens = max_new_tokens
        self.limiter = limiter or RateLimiter(0.25)
        self.logger = logger
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def infer(self, prompt: dict) -> str:
        def _call():
            self.limiter.acquire()

            messages = []
            if prompt.get('system'):
                messages.append({"role": "system", "content": prompt['system']})
            messages.append({"role": "user", "content": prompt['user']})

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/your-repo",  # Optional
                "X-Title": "Context-Memory-Agent"  # Optional
            }

            data = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_new_tokens,
                "temperature": 0.0
            }

            response = requests.post(self.base_url, headers=headers, json=data)
            response.raise_for_status()

            result = response.json()
            return result['choices'][0]['message']['content'].strip()

        return _retry_with_backoff(_call, self.logger)


class GeminiClient:
    def __init__(self, api_key: str, model_name: str, max_new_tokens: int = 512, limiter: RateLimiter | None = None, logger=None):
        self.genai = genai
        self.types = types
        self.client = genai.Client(api_key=api_key)
        self.model = model_name
        self.max_new_tokens = max_new_tokens
        self.limiter = limiter or RateLimiter(0.25)
        self.logger = logger

    def infer(self, prompt: dict) -> str:
        def _call():
            self.limiter.acquire()
            cfg = self.types.GenerateContentConfig(
                system_instruction=prompt['system'],
                max_output_tokens=self.max_new_tokens,
                temperature=0.0,
            )
            resp = self.client.models.generate_content(
                model=self.model,
                contents=prompt['user'],
                config=cfg,
            )
            return (resp.text or "").strip()
        return _retry_with_backoff(_call, self.logger)

class CalculatorTool:
    def calculate(self, expression: str):
        try:
            expr = expression.strip()
            if len(expr) > 512:
                return "Error"
            node = ast.parse(expr, mode="eval")
            return self._eval(node.body)
        except Exception:
            return "Error"
    def _eval(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = self._eval(node.operand)
            return +v if isinstance(node.op, ast.UAdd) else -v
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)):
            l = self._eval(node.left); r = self._eval(node.right)
            return {ast.Add: l+r, ast.Sub: l-r, ast.Mult: l*r, ast.Div: l/r,
                    ast.FloorDiv: l//r, ast.Mod: l%r, ast.Pow: l**r}[type(node.op)]
        if isinstance(node, ast.Name) and node.id in {"pi","e"}:
            return getattr(math, node.id)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"sin","cos","tan","sqrt","log","log10","exp","abs"}:
            fn = getattr(math, node.func.id) if node.func.id != "abs" else abs
            args = [self._eval(a) for a in node.args]
            return fn(*args)
        raise ValueError("invalid")

class SimpleLogger:
    def __init__(self, path="log.txt", truncate: bool = False):
        self.path = path
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        mode = "w" if truncate else "a"
        with open(self.path, mode, encoding="utf-8") as f:
            if truncate:
                f.write("")

    def log(self, text: str, ts: datetime.datetime | None = None):
        ts = ts or datetime.datetime.now()
        stamp = ts.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"{stamp} {text.rstrip()}\n")


class Agent:
    def __init__(self, gemini_client, tavily_client: TavilyClient, logger: SimpleLogger | None = None):
        # gemini_client parameter accepts both GeminiClient and OpenRouterClient
        self.gemini_client = gemini_client  # Keep name for backward compatibility
        self.llm_client = gemini_client  # Alias for clarity
        self.tavily_client = tavily_client
        self.calculator = CalculatorTool()
        self.logger = logger or SimpleLogger("log.txt")
        self.search_limiter = RateLimiter(0.4)
        self._search_cache = {}

    def _strip_code_fences(self, s: str) -> str:
        s = s.strip()
        s = re.sub(r"^```[\w-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
        s = s.replace("`", "").strip()
        return s

    def _format_history(self, history):
        parts = []
        for h in history:
            if h["tool"] == "calculator":
                parts.append(f"[calculator]\ninput: {h['input']}\noutput: {h['output']}")
            elif h["tool"] == "search":
                snip = []
                for r in h["output"]:
                    t = r.get("title",""); u = r.get("url",""); c = r.get("content","")
                    snip.append(f"- {t}\n  {u}\n  {c[:200]}")
                parts.append(f"[search]\ninput: {h['input']}\n" + "\n".join(snip))
        return "\n\n".join(parts).strip()

    def determine_if_calc_needed(self, question, history_text):
        prompt = {
            'system': "Respond with 'yes' or 'no' only.",
            'user': f"Question: {question}\nExisting tool outputs:\n{history_text if history_text else '(none)'}\nShould you perform a numeric calculation next? Answer yes or no."
        }
        return self.gemini_client.infer(prompt).strip().lower().rstrip(".") == "yes"

    def refine_calc_expression(self, question, history_text):
        prompt = {
            'system': "Transform the user question and given context into a single Python-style arithmetic expression. Output only the expression.",
            'user': f"Question: {question}\nContext:\n{history_text if history_text else '(none)'}"
        }
        expr = self.gemini_client.infer(prompt).strip()
        return self._strip_code_fences(expr)

    def determine_if_search_needed(self, question, history_text):
        prompt = {
            'system': "Respond with 'yes' or 'no' only.",
            'user': f"Question: {question}\nExisting tool outputs:\n{history_text if history_text else '(none)'}\nShould you search the internet next to answer this? Answer yes or no."
        }
        return self.gemini_client.infer(prompt).strip().lower().rstrip(".") == "yes"

    def refine_search_term(self, question, history_text):
        prompt = {
            'system': "Given a question and context, output a compact web search query (<=6 words). Output only the query.",
            'user': f"Question: {question}\nContext:\n{history_text if history_text else '(none)'}"
        }
        q = self.gemini_client.infer(prompt).strip()
        return self._strip_code_fences(q)

    def _search_call(self, query):
        def _call():
            self.search_limiter.acquire()
            return self.tavily_client.search(query=query, search_depth="basic", include_images=False)
        return _retry_with_backoff(_call, self.logger)

    def search_internet(self, query):
        key = query.strip().lower()
        if key in self._search_cache:
            return self._search_cache[key]
        res = self._search_call(query)
        self._search_cache[key] = res
        return res

    def generate_response(self, question, tool_history=None, user_context=None):
        tool_context = self._format_history(tool_history or [])

        # Build comprehensive prompt with user context
        system_prompt = "You are a helpful diet planning assistant. Use the provided context and tool outputs to answer the user's question."

        user_prompt_parts = []

        # Add user profile and history if available
        if user_context:
            user_prompt_parts.append("=== USER PROFILE ===")
            user_prompt_parts.append(user_context.get('profile_text', 'No profile information.'))
            user_prompt_parts.append("")

            if user_context.get('menu_history'):
                user_prompt_parts.append("=== RECENT MEALS (avoid repeating these) ===")
                user_prompt_parts.append(user_context['menu_history'])
                user_prompt_parts.append("")

            if user_context.get('recent_conversation'):
                user_prompt_parts.append("=== RECENT CONVERSATION ===")
                user_prompt_parts.append(user_context['recent_conversation'])
                user_prompt_parts.append("")

        # Add tool outputs
        if tool_context:
            user_prompt_parts.append("=== TOOL OUTPUTS ===")
            user_prompt_parts.append(tool_context)
            user_prompt_parts.append("")

        # Add current question
        user_prompt_parts.append("=== CURRENT QUESTION ===")
        user_prompt_parts.append(question)

        prompt = {
            'system': system_prompt,
            'user': "\n".join(user_prompt_parts)
        }

        self.logger.log("[final] prompt:\n" + prompt['user'] + "\n======================================================================\n")
        return self.gemini_client.infer(prompt)
    
    def chat_with_tools(self, question, max_steps=5, ts_start: datetime.datetime | None = None, user_context=None):
        self.logger.log(f"[start] question: {question}", ts=ts_start)
        history = []
        for i in range(1, max_steps + 1):
            history_text = self._format_history(history)
            self.logger.log(f"[iter {i}] question: {question}, history:\n{history_text if history_text else '(none)'}")
            need_calc = self.determine_if_calc_needed(question, history_text)
            if need_calc:
                expr = self.refine_calc_expression(question, history_text)
                self.logger.log(f"[iter {i}] calculator input: {expr}")
                result = self.calculator.calculate(expr)
                history.append({"tool":"calculator","input":expr,"output":result})
                continue
            history_text = self._format_history(history)
            need_search = self.determine_if_search_needed(question, history_text)
            if need_search:
                query = self.refine_search_term(question, history_text)
                self.logger.log(f"[iter {i}] search input: {query}")
                search_results = self.search_internet(query)
                trimmed = []
                for r in (search_results.get("results") or [])[:3]:
                    trimmed.append({"title": r.get("title",""), "url": r.get("url",""), "content": r.get("content","")})
                history.append({"tool":"search","input":query,"output":trimmed})
                continue
            break
        return self.generate_response(question, tool_history=history, user_context=user_context)


def get_agent_message(username: str, inquiry: str, timestamp: datetime.datetime, memory_manager=None) -> str:
    user_log_path = os.path.join("logs", f"{username}.log")
    logger = SimpleLogger(user_log_path, truncate=False)  # Don't truncate - keep persistent logs

    # Determine which LLM provider to use
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()  # Default to gemini

    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        model_name = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")
        llm_client = OpenRouterClient(
            api_key=api_key,
            model_name=model_name,
            max_new_tokens=1024,
            limiter=RateLimiter(0.3),
            logger=logger,
        )
    else:  # Default to Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        llm_client = GeminiClient(
            api_key=api_key,
            model_name=model_name,
            max_new_tokens=1024,
            limiter=RateLimiter(0.3),
            logger=logger,
        )

    tavily_api_key = os.getenv("TAVILY_API_KEY")
    tavily_client = TavilyClient(api_key=tavily_api_key)
    agent = Agent(gemini_client=llm_client, tavily_client=tavily_client, logger=logger)

    logger.log(f"[user] {username}", ts=timestamp)
    logger.log(f"[channel] discord", ts=timestamp)

    # Build user context from memory manager
    user_context = None
    if memory_manager:
        try:
            # Get context from memory manager
            user_context = memory_manager.build_context_for_llm(
                user_id=username,
                current_query=inquiry,
                include_menu_days=7,
                include_message_limit=10
            )

            # Add user message to history
            memory_manager.add_message(
                user_id=username,
                role='user',
                content=inquiry,
                timestamp=timestamp
            )

            logger.log(f"[context] Loaded profile and history for {username}")

        except Exception as e:
            logger.log(f"[warning] Failed to load user context: {repr(e)}")
            user_context = None

    try:
        # Generate response with context
        answer = agent.chat_with_tools(inquiry, max_steps=5, ts_start=timestamp, user_context=user_context)
        logger.log(f"[final] answer(len={len(answer)}): {answer[:500]}", ts=timestamp)

        # Save assistant response to history
        if memory_manager:
            try:
                memory_manager.add_message(
                    user_id=username,
                    role='assistant',
                    content=answer,
                    timestamp=datetime.datetime.now()
                )

                # Try to extract and update profile from this conversation
                conversation_text = f"User: {inquiry}\nAssistant: {answer}"
                extracted_profile = memory_manager.extract_profile_from_conversation(
                    user_id=username,
                    conversation_text=conversation_text,
                    gemini_client=llm_client
                )

                if extracted_profile:
                    memory_manager.update_user_profile(username, extracted_profile)
                    logger.log(f"[profile] Updated profile with: {list(extracted_profile.keys())}")

                # Extract and save menu items if this was a meal planning response
                try:
                    from menu_extractor import integrate_menu_tracking
                    meals = integrate_menu_tracking(
                        user_id=username,
                        response=answer,
                        gemini_client=llm_client,
                        memory_manager=memory_manager
                    )
                    if meals:
                        logger.log(f"[menu] Saved {len(meals)} meals to history")
                except Exception as e:
                    logger.log(f"[warning] Failed to extract menu items: {repr(e)}")

            except Exception as e:
                logger.log(f"[warning] Failed to save conversation: {repr(e)}")

        return answer

    except Exception as e:
        logger.log(f"[error] {repr(e)}", ts=timestamp)
        return "Sorry, something went wrong while generating the response."
    
def main():
    print(get_agent_message("user1", "If I invest $500 at 5% annual simple interest for 3 years, how much interest is earned?", datetime.datetime.now()))
    print(get_agent_message("user2", "What is the current status of the stock market?", datetime.datetime.now()))

if __name__ == "__main__":
    main()
