import argparse
import re


import genai_1_24 as tg


MIN_WORDS = 100
MAX_WORDS = 150


def count_words(text: str) -> int:
    return len(re.findall(r"[A-Za-zА-Яа-яЁё0-9']+", text))


def detect_letter_type(letter_type_arg: str | None) -> str:
    if not letter_type_arg:
        return "официальное"
    s = letter_type_arg.strip().lower()
    if s in ("оф", "официальное", "formal", "official"):
        return "официальное"
    if s in ("др", "дружеское", "friendly", "informal"):
        return "дружеское"
    return "официальное"


def build_draft_prompt(letter_type: str, topic: str, language: str = "ru") -> str:
    if language == "en":
        base = "Write an email draft."
        style = "formal business email" if letter_type == "официальное" else "friendly informal email"
        return f"{base} Topic: {topic}. Style: {style}. 110-130 words."
    else:
        style = "официально-деловой" if letter_type == "официальное" else "дружеский"
        return f"Напиши черновик письма на тему: {topic}. Стиль: {style}. Объём 110–130 слов."


def generate_draft(pipe, prompt: str, max_new_tokens: int = 220) -> str:
    messages = [
        {"role": "system", "content": "Ты пишешь короткие письма. Соблюдай ограничения по объёму и теме."},
        {"role": "user", "content": prompt},
    ]
    out = pipe(messages, max_new_tokens=max_new_tokens)[0]["generated_text"][-1]["content"].strip()
    return out


def format_final_letter(topic: str, body: str, signature: str, language: str = "ru") -> str:
    signature = signature.replace("\n", " ").strip()
    if language == "en":
        return f"Subject: {topic}\n\n{body}\n\n{signature}\n"
    return f"Тема: {topic}\n\n{body}\n\n{signature}\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default=None, help="официальное/дружеское (или оф/др)")
    parser.add_argument("--topic", default="Запрос информации", help="Тема письма")
    parser.add_argument("--sign", default="С уважением, Илья", help="Подпись")
    parser.add_argument("--lang", default="ru", choices=["ru", "en"], help="Язык письма")
    parser.add_argument("-o", "--output", default="output.txt", help="Файл .txt для сохранения")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507", help="Модель (как у одногруппника)")
    args = parser.parse_args()

    letter_type = detect_letter_type(args.type)

    pipe = tg.text_pipeline_init(args.model, padding="left")

    draft_prompt = build_draft_prompt(letter_type, args.topic, args.lang)
    draft = generate_draft(pipe, draft_prompt)

    style_for_tg = "official" if letter_type == "официальное" else "friendly"
    body = tg.run_style_transfer(pipe, draft, style=style_for_tg)

    tries = 6
    for _ in range(tries):
        w = count_words(body)
        if MIN_WORDS <= w <= MAX_WORDS:
            break
        # если мало — попросим расширить; если много — сократить
        if w < MIN_WORDS:
            body = tg.run_style_transfer(pipe, body + "\n\nСделай письмо чуть подробнее.", style=style_for_tg)
        else:
            body = tg.run_style_transfer(pipe, body + "\n\nСократи письмо до 100–150 слов.", style=style_for_tg)

    final_text = format_final_letter(args.topic, body, args.sign, args.lang)

    tg.write_text(args.output, final_text)

    print(f"Тип письма: {letter_type}")
    print(f"Слова: {count_words(body)}")
    print(f"Сохранено: {args.output}")


if __name__ == "__main__":
    main()
