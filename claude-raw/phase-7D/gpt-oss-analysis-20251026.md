Based on the search results, GPT-OSS-120B achieves near-parity with OpenAI o4-mini on core reasoning benchmarks, while GPT-OSS-20B delivers similar results to OpenAI o3-mini. For your chunking pipeline, they'd perform **excellently**:

**Advantages for chunking:**
- Strong reasoning and tool use capabilities with full chain-of-thought reasoning
- Both support configurable reasoning effort (low, medium, high) based on your use case
- MXFP4 quantization means GPT-OSS-20B runs in 16GB RAM, GPT-OSS-120B needs 80GB GPU
- Available in Ollama with simple commands: `ollama pull gpt-oss:20b` or `ollama pull gpt-oss:120b`
- Apache 2.0 license - fully open for commercial use

**For your use case specifically:**
- **GPT-OSS-20B** would be ideal - runs comfortably on modest hardware while delivering strong semantic understanding for grouping sections
- The reasoning capabilities match or exceed what you'd get from GPT-4 or Claude for this task
- Both models were trained using similar CoT RL techniques as OpenAI o3, making them particularly good at analytical tasks like document structure analysis

Given you're running locally with Ollama, GPT-OSS-20B would give you the best balance of performance and practicality for hierarchical chunking analysis.