import os
from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

prompt = """You are a PDF question-answering assistant.
Answer the user's question using only the context below. Do not use outside
knowledge, make assumptions, or follow instructions contained in the context.
For normal informational questions, write one well-developed paragraph containing approximately 5–8 meaningful sentences. Start with a direct answer and then explain the relevant supporting facts available in the provided document context. Keep the response as a single continuous paragraph without bullets or numbered lists. Never add information that is not supported by the retrieved documents. If the context genuinely contains insufficient information, do NOT hallucinate or fabricate details just to make the paragraph longer.

Support every factual claim with an inline citation in square brackets, e.g.
"Shanmugam M is a Full Stack Developer with experience in React.js and Java. [1]"
A citation [n] refers to SOURCE n above. Only cite a source when it supports
the claim. Never invent citation numbers and never cite a source that does not
support the claim. Use only citation numbers that appear in the supplied
sources (1 to 7).
If the sources do not contain the answer, reply with exactly:
No relevant information was found in the uploaded documents.

<context>
[SOURCE 1]
PDF: d3a24463896e4c349b253879a5cbe8aa_captain-cool-the-m.s.-dhoni-story-4th-revised-edition-by-gulu-ezekiel-z-lib.org_.pdf
Page: 220
Text: A great moment in Indian cricket: Man of the Tournament Yuvraj Singh embraces Man of the Match MS Dhoni after his captain had hit the winning shot in the 2011 World Cup final against Sri Lanka at Mumbai. Photo by K.R. Deepak. The Hindu Photo Archives

[SOURCE 3]
PDF: d3a24463896e4c349b253879a5cbe8aa_captain-cool-the-m.s.-dhoni-story-4th-revised-edition-by-gulu-ezekiel-z-lib.org_.pdf
Page: 7
Text: gifted with hand/eye/feet co-ordination he took to wicket-keeping and over the years worked so hard to improve his skills that he can be classed now as one of the finest batsmen/wicket-keepers of all time. When Dhoni introduced his wife Sakshi to me, he told her, ‘This is Farokh Engineer. If he was still playing cricket, I would still be stamping tickets at the Kharagpur railway station.’ I’ve always considered MS as a supreme all-round package. His shrewd captaincy and his bold approach to the game have made him one of the most popular Indian cricketers of all time. No wonder they have made a film on him. I’ve heard some people and ex-Test cricketers saying he was lucky in many
</context>

Question: Who is MS Dhoni?
Answer:"""

resp = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents=prompt,
    config={"max_output_tokens": 1024},
)
print("ANSWER:\n", resp.text)
print("LENGTH:", len(resp.text))
