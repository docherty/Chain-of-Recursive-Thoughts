# CoRT (Chain of Recursive Thoughts) 🧠🔄

## TL;DR: I made my AI think harder by making it argue with itself repeatedly. It works stupidly well.

### What is this?
CoRT makes AI models recursively think about their responses, generate alternatives, and pick the best one. It's like giving the AI the ability to doubt itself and try again... and again... and again, using the OpenRouter API.

### Does it actually work?
YES. I tested it with Mistral 7B Instruct (free tier) and it went from "meh" to "holy crap", especially for such a small model, at programming tasks. (Performance varies by model).

## How it works
1.  User provides input.
2.  AI (via the script) estimates how many "thinking rounds" it needs based on the input's complexity.
3.  AI generates an initial response.
4.  For each thinking round:
    *   Generates alternative responses (e.g., 2 alternatives).
    *   Evaluates the current best response against the alternatives.
    *   Picks the best one (which might be the current best or an alternative).
5.  The final response is the survivor of this iterative refinement process.

## Examples
*(You might need to update these image links if you host them directly in your repository)*

Mistral 7B Instruct *without* CoRT (Example Output):
![non-rec](https://github.com/user-attachments/assets/9c4f6af9-0a8f-4c62-920c-f272fce225c1)

Mistral 7B Instruct *with* CoRT (Example Output):
![rec](https://github.com/user-attachments/assets/acbcf1f9-4715-4d2c-a31c-38b349602380)


## Try it yourself
1.  **Clone/Download:** Get the code.
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set API Key:**
    ```bash
    # Recommended: Set environment variable (Linux/macOS)
    export OPENROUTER_API_KEY="your-key-here"
    # Or Windows (Command Prompt)
    # set OPENROUTER_API_KEY="your-key-here"
    # Or Windows (PowerShell)
    # $env:OPENROUTER_API_KEY="your-key-here"
    ```
    *Alternatively, the script will prompt you if the environment variable is not found.*
4.  **Run the Script:**
    ```bash
    python recursive_thinking_ai.py
    ```
5.  **Interact:** Chat with the AI!
    *   Type `save` to save the current conversation history.
    *   Type `save log` to save the detailed thinking log for all interactions in the session.
    *   Type `exit` to quit.

### The Secret Sauce
The magic is in:
*   **Self-Correction/Refinement:** The core loop of generating, evaluating, and selecting.
*   **Competitive Alternatives:** Forcing the generation of different approaches.
*   **Iterative Improvement:** Building on the best response from the previous round.
*   **Dynamic Thinking Depth:** Adapting the number of rounds based on perceived complexity.

### Contributing
Found a way to make it even better? Ideas for improvement? Pull Requests and Issues are welcome!

### License
MIT - Go wild with it.