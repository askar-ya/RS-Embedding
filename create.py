from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

token = os.getenv('TOKEN')

# loads model with CLS pooling
model = SentenceTransformer("ai-forever/FRIDA", token=token)

#query_embedding = model.encode("Сколько программистов нужно, чтобы вкрутить лампочку?", prompt_name="search_query")
document_embedding = model.encode("Чтобы вкрутить лампочку, требуется три программиста: один напишет программу извлечения лампочки, другой — вкручивания лампочки, а третий проведет тестирование.", prompt_name="search_document")
print(document_embedding)
#print(query_embedding @ document_embedding.T) # 0.7285831
