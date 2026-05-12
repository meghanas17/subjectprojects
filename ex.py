import wikipediaapi
import pandas as pd
import re
import time
import random

languages = {
    "en": {"name": "English", "topics": ["India", "Science", "Technology", "Education", "Culture"]},
    "kn": {"name": "Kannada", "topics": ["ಭಾರತ", "ವಿಜ್ಞಾನ", "ತಂತ್ರಜ್ಞಾನ", "ಶಿಕ್ಷಣ", "ಸಂಸ್ಕೃತಿ"]},
    "ta": {"name": "Tamil", "topics": ["இந்தியா", "அறிவியல்", "தொழில்நுட்பம்", "கல்வி", "கலை"]},
    "hi": {"name": "Hindi", "topics": ["भारत", "विज्ञान", "इतिहास", "संस्कृति", "प्रौद्योगिकी", "शिक्षा", "भारत_का_इतिहास"]}
}

USER_AGENT = "MiniLangCollector/1.0 (contact: your_email@example.com)"
data = []

for lang_code, info in languages.items():
    wiki = wikipediaapi.Wikipedia(language=lang_code, user_agent=USER_AGENT)
    sentences = []

    print(f"\n🔍 Collecting {info['name']} sentences...")

    for topic in info["topics"]:
        try:
            page = wiki.page(topic)
            if not page.exists():
                continue

            text = re.sub(r'\n+', ' ', page.text)
            sents = re.split(r'(?<=[.!?।]) +', text)  # added Hindi full stop '।'

            for s in sents:
                if 10 <= len(s) <= 200:
                    sentences.append(s.strip())
                if len(sentences) >= 100:
                    break
            if len(sentences) >= 100:
                break
            time.sleep(2)

        except Exception as e:
            print(f"⚠️ Error fetching topic '{topic}' for {info['name']}: {e}")
            continue

    if len(sentences) == 0:
        print(f"⚠️ No sentences found for {info['name']}!")
    else:
        print(f"✅ Collected {len(sentences)} {info['name']} sentences.")

    for s in sentences:
        data.append([s, info["name"]])

df = pd.DataFrame(data, columns=['text', 'language'])
df.to_csv("mini_multilingual.csv", index=False, encoding='utf-8')

print("\n✅ Dataset created successfully with shape:", df.shape)
 