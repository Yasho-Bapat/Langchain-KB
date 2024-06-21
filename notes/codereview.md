# code review notes for Ask Viridium AI

- added two examples in openai playground. (apart from system message) given as **user-assistant pairs**. 
- preparing a list of examples to send to the llm
- LanguageModel class used to connect to some LLM (openai rn, but trying to expand to gemini also)
- languagemodel.completechat(materialname, prompt, examples, temperature)
- using basic api, no sdk. GET and POST requests
- no changes/post-processing done to the payload recd by backend while passing to frontend
- sending name, manufacturer