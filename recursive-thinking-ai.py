import os
import json
import requests
import re
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import time # Keep for potential future use (e.g., rate limiting)
from dotenv import load_dotenv

load_dotenv()

class EnhancedRecursiveThinkingChat:
    """
    A chat client that uses a recursive thinking process involving multiple rounds
    of generation, alternative creation, and evaluation via an LLM API (OpenRouter).
    """
    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "mistralai/mistral-7b-instruct:free", # Updated default model example
                 base_url: str = "https://openrouter.ai/api/v1",
                 request_timeout: int = 60,
                 default_max_tokens: int = 4096,
                 save_directory: str = "chats"): # Added save directory parameter
        """
        Initialize the chat client.

        Args:
            api_key: Your OpenRouter API key. Reads from OPENROUTER_API_KEY env var if None.
            model: The model identifier to use on OpenRouter.
            base_url: The base URL for the OpenRouter API.
            request_timeout: Timeout in seconds for API requests.
            default_max_tokens: Default maximum tokens for API responses.
            save_directory: The subdirectory name to save chat logs and history.
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("API key not provided and OPENROUTER_API_KEY environment variable not set.")

        self.model = model
        self.base_url = base_url
        self.request_timeout = request_timeout
        self.default_max_tokens = default_max_tokens
        self.chat_completions_url = f"{self.base_url}/chat/completions"
        self.save_directory = save_directory # Store save directory

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # Optional: Add Referer/X-Title if required by your setup or OpenRouter TOS
            # "HTTP-Referer": "YOUR_SITE_URL",
            # "X-Title": "YOUR_APP_TITLE",
        }
        self.conversation_history: List[Dict[str, str]] = []
        self.full_thinking_log: List[Dict[str, Any]] = [] # Stores thinking history from all calls

    def _call_api(self,
                  messages: List[Dict[str, str]],
                  temperature: float = 0.7,
                  max_tokens: Optional[int] = None,
                  stream: bool = True) -> str:
        """
        Make an API call to the OpenRouter Chat Completions endpoint.

        Args:
            messages: A list of message dictionaries (e.g., [{"role": "user", "content": "..."}]).
            temperature: The sampling temperature.
            max_tokens: Maximum tokens to generate. Uses instance default if None.
            stream: Whether to stream the response.

        Returns:
            The complete response content as a string.

        Raises:
            requests.exceptions.RequestException: If the API request fails.
            ValueError: If the response format is unexpected.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self.default_max_tokens,
            "stream": stream,
        }

        try:
            response = requests.post(
                self.chat_completions_url,
                headers=self.headers,
                json=payload,
                stream=stream,
                timeout=self.request_timeout
            )
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

            if stream:
                full_response = ""
                print("AI Stream: ", end="", flush=True)
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith("data: "):
                            content = decoded_line[6:]
                            if content.strip() == "[DONE]":
                                break
                            try:
                                chunk = json.loads(content)
                                if chunk.get("choices"):
                                    delta = chunk["choices"][0].get("delta", {})
                                    content_piece = delta.get("content")
                                    if content_piece:
                                        full_response += content_piece
                                        print(content_piece, end="", flush=True)
                            except json.JSONDecodeError:
                                print(f"\nWarning: Could not decode JSON stream chunk: {content}", flush=True)
                                continue # Ignore malformed chunks
                print() # Newline after streaming
                return full_response
            else:
                # Non-streaming response
                response_data = response.json()
                if response_data.get("choices"):
                    return response_data['choices'][0]['message']['content'].strip()
                else:
                    raise ValueError(f"Unexpected API response format: {response_data}")

        except requests.exceptions.RequestException as e:
            print(f"\nAPI Request Error: {e}", flush=True)
            raise # Re-raise the exception for handling upstream
        except Exception as e:
            print(f"\nError processing API response: {e}", flush=True)
            # Return error message or raise a custom exception
            return f"Error: Could not process response: {e}"


    def _determine_thinking_rounds(self, prompt: str) -> int:
        """
        Asks the LLM to estimate the optimal number of thinking rounds (1-5).

        Args:
            prompt: The user's input prompt.

        Returns:
            The estimated number of rounds (1-5), defaulting to 3 on failure.
        """
        meta_prompt = f"""Analyze the following user message:
"{prompt}"

Based on its apparent complexity, ambiguity, and the detail likely required for a high-quality response, estimate the optimal number of iterative thinking rounds needed (between 1 and 5). A simple factual query might need 1 round, while a complex creative or analytical task might need 3-5.

Respond ONLY with a single digit (1, 2, 3, 4, or 5)."""

        messages = [{"role": "user", "content": meta_prompt}]
        default_rounds = 3

        print("\n=== DETERMINING THINKING ROUNDS ===")
        try:
            # Use lower temperature for more deterministic choice
            response = self._call_api(messages, temperature=0.2, stream=False)
            print(f"LLM estimated rounds response: '{response}'") # Show raw response for debugging

            # Robust parsing using regex
            match = re.search(r'\b([1-5])\b', response)
            if match:
                rounds = int(match.group(1))
                print(f"Parsed rounds: {rounds}")
                return rounds
            else:
                print(f"Could not parse rounds from response, using default: {default_rounds}")
                return default_rounds
        except Exception as e:
            print(f"Error determining thinking rounds: {e}. Using default: {default_rounds}")
            return default_rounds
        finally:
            print("=" * 30 + "\n")


    def _generate_alternatives(self, base_response: str, prompt: str, round_num: int, num_alternatives: int = 2) -> List[str]:
        """
        Generates alternative responses based on the current best response.

        Args:
            base_response: The current best response.
            prompt: The original user prompt.
            round_num: The current thinking round number.
            num_alternatives: How many alternatives to generate.

        Returns:
            A list of alternative response strings.
        """
        alternatives = []
        # Include conversation history for context
        context_messages = self.conversation_history + [{"role": "user", "content": prompt}]

        for i in range(num_alternatives):
            print(f"\n--- Generating Alternative {i+1} (Round {round_num}) ---")
            alt_prompt = f"""The original user message was: "{prompt}"

We are in thinking round {round_num}. The current best response is:
"{base_response}"

Generate a significantly different alternative response. Consider alternative interpretations, structures, or levels of detail. Focus on improving clarity, accuracy, or helpfulness compared to the current best. Do not simply rephrase; offer a distinct approach.

Alternative response:"""

            # Slightly increase temperature for diversity in alternatives
            temp = 0.7 + (i * 0.1)
            try:
                # Pass only the user alt_prompt for this specific task
                alternative = self._call_api(
                    context_messages + [{"role": "assistant", "content": base_response}, {"role": "user", "content": alt_prompt}],
                    temperature=temp,
                    stream=True
                )
                alternatives.append(alternative)
            except Exception as e:
                print(f"Error generating alternative {i+1}: {e}")
                alternatives.append(f"Error generating alternative {i+1}") # Add placeholder
            print("-" * 30)

        return alternatives


    def _evaluate_responses(self, prompt: str, current_best: str, alternatives: List[str]) -> Tuple[str, str]:
        """
        Asks the LLM to evaluate the current best response against alternatives.

        Args:
            prompt: The original user prompt.
            current_best: The current best response string.
            alternatives: A list of alternative response strings.

        Returns:
            A tuple containing (selected_best_response, explanation).
            Returns (current_best, "Evaluation failed") on error.
        """
        print("\n=== EVALUATING RESPONSES ===")
        eval_prompt = f"""Original user message:
"{prompt}"

Evaluate the following responses based on accuracy, clarity, completeness, and relevance to the original message.

Current Best Response:
"{current_best}"

Alternative Responses:
{chr(10).join([f"{i+1}. {alt}" for i, alt in enumerate(alternatives)])}

Which response is the overall best?

Instructions:
1.  First line: Respond ONLY with the word 'current' or the number of the best alternative (1-{len(alternatives)}).
2.  Second line: Provide a brief (1-2 sentence) explanation for your choice.

Example Response:
current
This response was the most direct and accurate.

Example Response:
2
Alternative 2 offered a clearer step-by-step explanation.

Your evaluation:"""

        messages = [{"role": "user", "content": eval_prompt}]
        default_choice = current_best
        default_explanation = "Defaulted to current best due to evaluation error."

        try:
            # Use low temperature for consistent evaluation
            evaluation = self._call_api(messages, temperature=0.1, stream=False)
            print(f"LLM evaluation response:\n---\n{evaluation}\n---")

            # Robust parsing using regex
            match = re.match(r"^\s*(current|\d+)\s*\n?(.*)", evaluation, re.DOTALL | re.IGNORECASE)

            if match:
                choice_str = match.group(1).lower().strip()
                explanation = match.group(2).strip() if match.group(2) else "No explanation provided."

                if choice_str == 'current':
                    print(f"Evaluation result: Keep Current. Reason: {explanation}")
                    return current_best, explanation
                else:
                    try:
                        choice_idx = int(choice_str) - 1
                        if 0 <= choice_idx < len(alternatives):
                            print(f"Evaluation result: Select Alternative {choice_idx + 1}. Reason: {explanation}")
                            return alternatives[choice_idx], explanation
                        else:
                            print(f"Warning: Invalid alternative number '{choice_str}' received.")
                    except ValueError:
                        print(f"Warning: Could not parse choice number '{choice_str}'.")
            else:
                 print("Warning: Could not parse evaluation response format.")

        except Exception as e:
            print(f"Error during evaluation: {e}")
        finally:
            print("=" * 30 + "\n")
        # Fallback on error
        print(f"Evaluation failed, keeping current best.")
        return default_choice, default_explanation


    def think_and_respond(self, user_input: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Processes user input using the recursive thinking workflow.

        Args:
            user_input: The user's message.
            verbose: If True, prints detailed thinking steps.

        Returns:
            A dictionary containing the final response and thinking history:
            {
                "response": str,
                "thinking_rounds": int,
                "thinking_history": List[Dict]
            }
        """
        print("\n" + "=" * 50)
        print("🤔 STARTING RECURSIVE THINKING PROCESS")
        print(f"Original User Input: {user_input}")
        print("=" * 50)

        # 1. Determine rounds needed
        thinking_rounds = self._determine_thinking_rounds(user_input)
        if verbose:
            print(f"\n🧠 Estimated Thinking Rounds: {thinking_rounds}")

        # 2. Initial response generation
        print("\n=== GENERATING INITIAL RESPONSE (Round 0) ===")
        # Add user input to history temporarily for this call, but commit later
        initial_messages = self.conversation_history + [{"role": "user", "content": user_input}]
        try:
            current_best = self._call_api(initial_messages, stream=True)
        except Exception as e:
             return {
                "response": f"Error generating initial response: {e}",
                "thinking_rounds": 0,
                "thinking_history": [{"round": 0, "error": str(e)}]
            }
        print("=" * 30)

        # Initialize thinking history for this call
        call_thinking_history = [{"round": 0, "response": current_best, "selected": True, "explanation": "Initial response"}]

        # 3. Iterative improvement rounds
        for round_num in range(1, thinking_rounds + 1):
            if verbose:
                print(f"\n=== STARTING THINKING ROUND {round_num}/{thinking_rounds} ===")

            # 3a. Generate alternatives
            alternatives = self._generate_alternatives(current_best, user_input, round_num)
            if not alternatives:
                 if verbose: print("No alternatives generated, skipping evaluation for this round.")
                 continue # Skip evaluation if generation failed

            # Add alternatives to this call's history
            for i, alt in enumerate(alternatives):
                call_thinking_history.append({
                    "round": round_num,
                    "response": alt,
                    "selected": False,
                    "alternative_number": i + 1
                })

            # 3b. Evaluate and select best
            new_best, explanation = self._evaluate_responses(user_input, current_best, alternatives)

            # Update selection in this call's history and set current_best
            if new_best != current_best:
                 # Find the selected alternative in history and mark it
                 found = False
                 for item in call_thinking_history:
                     if item["round"] == round_num and item["response"] == new_best and not item["selected"]:
                         item["selected"] = True
                         item["explanation"] = explanation
                         found = True
                         break
                 if not found:
                     # Handle case where evaluation returned something unexpected
                     print(f"Warning: Selected response not found in round {round_num} alternatives.")
                     # Add it anyway, marking it selected
                     call_thinking_history.append({
                         "round": round_num,
                         "response": new_best,
                         "selected": True,
                         "explanation": explanation + " (Added post-evaluation)",
                         "alternative_number": -1 # Indicate it wasn't a generated alternative
                     })

                 current_best = new_best
                 if verbose: print(f"✅ Round {round_num}: New best selected. Reason: {explanation}")

            else:
                 # Mark the previously selected item (could be from round 0 or previous rounds)
                 updated_previous = False
                 for item in reversed(call_thinking_history):
                     if item["selected"] and item["response"] == current_best:
                         item["explanation"] = explanation # Update explanation on the existing best
                         updated_previous = True
                         break
                 if not updated_previous:
                     print(f"Warning: Could not find item to attach 'keep current' explanation for round {round_num}")

                 if verbose: print(f"✅ Round {round_num}: Kept current best. Reason: {explanation}")

            if verbose: print(f"=== COMPLETED THINKING ROUND {round_num}/{thinking_rounds} ===")


        # 4. Finalize
        # Add user input and final assistant response to persistent history
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": current_best})

        # Keep conversation history manageable (optional: implement token limit instead)
        history_limit = 10 # Keep last 5 pairs
        if len(self.conversation_history) > history_limit:
            self.conversation_history = self.conversation_history[-history_limit:]

        # Append this call's thinking process to the full log
        self.full_thinking_log.append({
            "user_input": user_input,
            "final_response": current_best,
            "thinking_rounds_executed": thinking_rounds,
            "history": call_thinking_history,
            "timestamp": datetime.now().isoformat()
        })

        print("\n" + "=" * 50)
        print("🎯 RECURSIVE THINKING PROCESS COMPLETE")
        print("=" * 50)

        return {
            "response": current_best,
            "thinking_rounds": thinking_rounds,
            "thinking_history": call_thinking_history # Return history for this specific call
        }

    def _save_json(self, data: Any, filename: str, description: str):
        """Helper to save data to a JSON file in the designated subdirectory."""
        try:
            # Ensure the save directory exists
            os.makedirs(self.save_directory, exist_ok=True)

            # Construct the full path
            full_path = os.path.join(self.save_directory, filename)

            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"{description} saved to {full_path}")
        except IOError as e:
            print(f"Error saving {description} to {full_path}: {e}")
        except TypeError as e:
            print(f"Error serializing {description} data for saving: {e}")
        except Exception as e: # Catch other potential errors like permission issues
             print(f"An unexpected error occurred while saving {description} to {full_path}: {e}")


    def save_full_log(self, filename: Optional[str] = None):
        """Save the accumulated thinking process log from all calls."""
        if not self.full_thinking_log:
            print("No thinking log entries to save.")
            return

        if filename is None:
            filename = f"full_thinking_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        log_data = {
            "session_model": self.model,
            "session_start_time": self.full_thinking_log[0]['timestamp'] if self.full_thinking_log else "N/A",
            "log_save_time": datetime.now().isoformat(),
            "full_log": self.full_thinking_log # Save the accumulated log
        }
        self._save_json(log_data, filename, "Full thinking log")


    def save_conversation(self, filename: Optional[str] = None):
        """Save the main conversation history."""
        if not self.conversation_history:
            print("No conversation history to save.")
            return

        if filename is None:
            filename = f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json" # Renamed for clarity

        convo_data = {
             "session_model": self.model,
             "log_save_time": datetime.now().isoformat(),
             "conversation": self.conversation_history
        }
        self._save_json(convo_data, filename, "Conversation history")


def display_thinking_history(history: List[Dict[str, Any]]):
    """Formats and prints the thinking history for a single call."""
    print("\n--- DETAILED THINKING PROCESS ---")
    if not history:
        print("No thinking history recorded for this call.")
        return

    last_round = -1
    for item in history:
        round_num = item.get("round", "N/A")
        if round_num != last_round:
            print(f"\n--- Round {round_num} ---")
            last_round = round_num

        is_selected = item.get("selected", False)
        is_initial = (round_num == 0)
        alt_num = item.get("alternative_number")

        prefix = ""
        if is_selected:
            prefix = "✅ [SELECTED]"
        elif alt_num is not None:
            prefix = f"   [Alternative {alt_num}]"
        elif not is_initial :
             prefix = "   [Intermediate]" # Fallback label

        print(f"{prefix}")
        # Indent response for clarity
        response_lines = item.get('response', 'N/A').split('\n')
        print(f"  Response: {response_lines[0]}")
        for line in response_lines[1:]:
            print(f"            {line}")

        explanation = item.get("explanation")
        if is_selected and explanation:
            print(f"  Reason: {explanation}")
        # print("-" * 20) # Optional separator between items

    print("--- END OF THINKING PROCESS ---")


def main():
    print("🤖 Enhanced Recursive Thinking Chat Initializing...")
    print("=" * 50)

    # Get API key - prefer environment variable
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("⚠️ OPENROUTER_API_KEY environment variable not found.")
        api_key = input("Enter your OpenRouter API key: ").strip()
        if not api_key:
            print("❌ Error: No API key provided. Exiting.")
            return
    else:
        print("🔑 API key loaded from environment variable.")

    # Choose model (optional)
    default_model = "mistralai/mistral-7b-instruct:free" # Example free model
    model_choice = input(f"Enter model name (or press Enter for default: {default_model}): ").strip()
    model = model_choice if model_choice else default_model
    print(f"Using model: {model}")

    # Define save directory
    save_dir = "chats"
    print(f"Logs and history will be saved to the '{save_dir}/' directory.")

    # Initialize chat
    try:
        # Pass the save directory to the constructor
        chat = EnhancedRecursiveThinkingChat(api_key=api_key, model=model, save_directory=save_dir)
    except ValueError as e:
        print(f"❌ Error initializing chat: {e}")
        return
    except Exception as e:
         print(f"❌ An unexpected error occurred during initialization: {e}")
         return


    print("\nChat ready! Type 'exit' to quit, 'save' to save conversation, 'save log' to save the full thinking log.")
    print("The AI will perform recursive thinking before responding.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError: # Handle Ctrl+D
             print("\nEOF detected, exiting.")
             break

        if not user_input:
            continue

        lower_input = user_input.lower()

        if lower_input == 'exit':
            break
        elif lower_input == 'save':
            chat.save_conversation()
            continue
        elif lower_input == 'save log':
            chat.save_full_log()
            continue

        # Get response with thinking process
        try:
            result = chat.think_and_respond(user_input, verbose=True) # Keep verbose for demo

            print(f"\n{'='*20} FINAL AI RESPONSE {'='*20}")
            print(result['response'])
            print(f"{'='* (40 + len(' FINAL AI RESPONSE '))}\n")

            # Option to display the detailed thinking steps
            show_thinking = input("Show detailed thinking steps for this response? (y/n, default n): ").strip().lower()
            if show_thinking == 'y':
                 display_thinking_history(result['thinking_history'])

        except requests.exceptions.RequestException as e:
             print(f"\n❌ Network/API Error: {e}. Please check connection/API key and try again.")
        except Exception as e:
             print(f"\n❌ An unexpected error occurred: {e}")
             # Optionally: save state before potentially crashing further
             # chat.save_conversation("error_dump_convo.json")
             # chat.save_full_log("error_dump_log.json")

    # Save on exit?
    print("\nExiting chat.")
    if chat.conversation_history: # Only ask if there's something to save
        save_on_exit = input("Save conversation before exiting? (y/n): ").strip().lower()
        if save_on_exit == 'y':
            chat.save_conversation()

    if chat.full_thinking_log:
        save_full = input("Save full thinking log before exiting? (y/n): ").strip().lower()
        if save_full == 'y':
            chat.save_full_log()

    print("\nGoodbye! 👋")

if __name__ == "__main__":
    main()