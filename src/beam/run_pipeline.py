import json
import os
import pickle
import multiprocessing as mp
import argparse
import time

from src.beam.main import generate_plans, add_special_bullets_to_plan, get_labels, get_profile, user_messages_generation_fast, user_messages_generation_parallel, answer_generation
from src.beam.ten_milion_pipeline import ten_m_plan_generation, ten_m_user_messages_generation, ten_m_answer_generation
from src.beam.generation_settings import get_plan_settings, get_question_settings, get_token_limit, normalize_domain
from src.llm import BuildLLm


def get_chat_directory_for_index(chats_directory: str, index: int) -> str:
    """Resolve an index through topics.json so partial/resumed runs stay aligned."""
    topics_address = os.path.join(chats_directory, "topics.json")
    with open(topics_address, "r", encoding="utf-8") as f:
        topics = json.load(f)
    try:
        chat_id = str(topics[index]["id"])
    except IndexError as exc:
        raise IndexError(
            f"Topic index {index} is outside the {len(topics)} configured topics"
        ) from exc
    return os.path.join(chats_directory, chat_id)


def run_plans_generation(chats_directory: str,
                       chat_size: str,
                       index: str):

    topics_address = os.path.join(chats_directory, "topics.json")
    with open(topics_address, "r", encoding="utf-8") as f:
        data = json.load(f)

    object = data[index]

    id = str(object['id'])

    chat_address = os.path.join(chats_directory, id)
    os.makedirs(chat_address, exist_ok=True)

    chat_topic_address = os.path.join(chat_address, "topic.json")
    with open(chat_topic_address, "w", encoding="utf-8") as f:
        json.dump(object, f, indent=4)

    sub_categories = ", ".join(object["subtopics"])
    category = object["category"]
    title = object["title"]
    theme = object["theme"]
    if "timeline" in object.keys():
        timeline = object["timeline"]
    else:
        timeline = "N/A"

    if "mode" in object.keys():
        mode = object["mode"]
    else:
        mode = "similar"

    if "num_plans" in object.keys():
        NUM_PLANS = object["num_plans"]
    else:
        NUM_PLANS = 10

    topic = category + " -> " + title
    theme = theme + " -> " + sub_categories

    category = normalize_domain(category)

    if chat_size != "10M":
        labels_address = os.path.join(chat_address, "labels.txt")
        main_spec_address = os.path.join(chat_address, "main_spec.txt")
        relationships_address = os.path.join(
            chat_address, "relationships.txt")
        if all(os.path.isfile(path) for path in (
            labels_address, main_spec_address, relationships_address
        )):
            with open(labels_address, encoding="utf-8") as f:
                labels = f.read()
            with open(main_spec_address, encoding="utf-8") as f:
                main_spec = f.read()
            with open(relationships_address, encoding="utf-8") as f:
                relationships = f.read()
            print("Reusing checkpointed labels and user profile")
        else:
            labels = get_labels(topic=topic, theme=theme, domain=category)
            main_spec, relationships = get_profile()
            with open(labels_address, "w", encoding="utf-8") as f:
                f.write(labels)
            with open(main_spec_address, "w", encoding="utf-8") as f:
                f.write(main_spec)
            with open(relationships_address, "w", encoding="utf-8") as f:
                f.write(relationships)

    NUM_BATCHES, NUM_BULLETS = get_plan_settings(chat_size, category)

    if chat_size == "10M":
        ten_m_plan_generation(initial_topic=object,
                              first_timeline=timeline,
                              num_plans=NUM_PLANS,
                              num_batches=str(NUM_BATCHES),
                              domain=category,
                              save_address=chat_address,
                              llm_name="gpt",
                              mode=mode)

        entries = os.listdir(chat_address)
        dirs = sorted(
            [name for name in entries if os.path.isdir(
                os.path.join(chat_address, name))],
            key=lambda x: int(x.split('-')[1])
        )

        for dir in dirs:
            plan_directory = os.path.join(chat_address, dir, "plan")

            add_special_bullets_to_plan(plan_address=plan_directory,
                                        num_bullets=NUM_BULLETS,
                                        llm_name="gpt")

    else:
        plan_save_address = os.path.join(chat_address, "plan")
        generate_plans(topic=topic, theme=theme, timeline=timeline, num_batches=NUM_BATCHES, num_bullets=NUM_BULLETS,
                       labels=labels, main_spec=main_spec, relationships=relationships,
                       llm_name="gpt", save_address=plan_save_address, input_address=None, domain=category)

        add_special_bullets_to_plan(plan_address=plan_save_address,
                                    num_bullets=NUM_BULLETS,
                                    llm_name="gpt")


def run_question_generation(chats_directory: str,
                            chat_size: str,
                            index: int,
                            question_workers: int = 1):

    chat_directory = get_chat_directory_for_index(chats_directory, index)

    topic_address = os.path.join(chat_directory, "topic.json")
    with open(topic_address, "r", encoding="utf-8") as f:
        data = json.load(f)

    title = data['title']
    theme = data['theme']
    category = data["category"]
    sub_categories = ", ".join(data["subtopics"])
    topic = category + " -> " + title
    theme = theme + " -> " + sub_categories

    topic = topic
    theme = theme
    category = normalize_domain(category)

    NUM_BATCHES, BATCH_SIZE, SUB_BATCHES_PER_BATCH = get_question_settings(
        chat_size, category
    )

    SUB_BATCH_SIZE = BATCH_SIZE // SUB_BATCHES_PER_BATCH

    if chat_size != "10M":
        plan_address = os.path.join(chat_directory, "plan_new.pickle")
        with open(plan_address, 'rb') as f:
            plan = pickle.load(f)

        save_address = os.path.join(chat_directory, "user_messages.pickle")
        if question_workers > 1:
            return user_messages_generation_parallel(
                topic=topic,
                theme=theme,
                plans=plan,
                num_batches=NUM_BATCHES,
                sub_batches_per_batch=SUB_BATCHES_PER_BATCH,
                sub_batch_size=SUB_BATCH_SIZE,
                batch_size=BATCH_SIZE,
                llm_name="llama",
                save_address=save_address,
                domain=category,
                sepcial_bullets=True,
                max_workers=question_workers,
            )
        user_messages_generation_fast(topic=topic, theme=theme, plans=plan, num_batches=NUM_BATCHES,
                                      sub_batches_per_batch=SUB_BATCHES_PER_BATCH,
                                      sub_batch_size=SUB_BATCH_SIZE, batch_size=BATCH_SIZE,
                                      llm_name="llama", save_address=save_address,
                                      domain=category, sepcial_bullets=True)
        return {"mode": "sequential", "max_workers": 1}
    else:
        ten_m_user_messages_generation(plan_save_address=chat_directory,
                                       num_batches=NUM_BATCHES,
                                       sub_batches_per_batch=SUB_BATCHES_PER_BATCH,
                                       sub_batch_size=SUB_BATCH_SIZE,
                                       batch_size=BATCH_SIZE,
                                       llm_name="llama",
                                       domain=category,
                                       mode="parallel")


def run_answer_generation(chats_directory: str,
                          chat_size: str,
                          index: int,
                          llm,
                          answer_max_tokens: int = 1_000):

    chat_directory = get_chat_directory_for_index(chats_directory, index)

    topic_address = os.path.join(chat_directory, "topic.json")
    with open(topic_address, "r", encoding="utf-8") as f:
        data = json.load(f)

    title = data['title']
    theme = data['theme']
    category = data["category"]
    sub_categories = ", ".join(data["subtopics"])
    topic = category + " -> " + title
    theme = theme + " -> " + sub_categories

    topic = topic
    theme = theme
    category = normalize_domain(category)

    if chat_size != "10M":
        messages_address = os.path.join(
            chat_directory, "user_messages.pickle")
        output_address = os.path.join(chat_directory, "chat.pickle")
        plan_address = os.path.join(chat_directory, "plan_new.pickle")
        
        answer_generation(
            input_address=messages_address,
            output_address=output_address,
            plans_address=plan_address,
            topic=topic,
            theme=theme,
            llm=llm,
            stop_token_target=64_000 if chat_size == "64K" else None,
            stop_token_limit=get_token_limit("64K") if chat_size == "64K" else None,
            tokenizer_model="meta-llama/llama-3.1-8b-instruct",
            response_max_tokens=answer_max_tokens,
        )
    else:
        ten_m_answer_generation(input_directory=chat_directory,
                                mode="parallel",
                                llm=llm)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process answer generation parameters")

    parser.add_argument("--model_url", type=str,
                        required=True, help="URL of the VLLM server")
    parser.add_argument("--model_name", type=str,
                        required=True, help="Directory of the LLM")
    parser.add_argument("--api_key", type=str, required=True,
                        help="API key for authentication")
    parser.add_argument("--generation_stage", type=str, required=True,
                        help="Generation Stage", choices=["plan", "question", "answer"])
    parser.add_argument("--start_index", type=int,
                        required=True, help="Starting index")
    parser.add_argument("--end_index", type=int,
                        required=True, help="Ending index")
    parser.add_argument("--chats_dir", type=str, required=True,
                        help="Directory containing chats")
    parser.add_argument("--chat_size", type=str,
                        required=True, help="Size of chat")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    english_only_regex = "[\\u0000-\\u2E7F]+"

    print("Model URL:", args.model_url)
    print("Model Name:", args.model_name)
    print("API Key: [redacted]")
    print("Generation Stage:", args.generation_stage)
    print("Start Index:", args.start_index)
    print("End Index:", args.end_index)
    print("Chats Directory:", args.chats_dir)
    print("Chat Size:", args.chat_size)

    qwen_awq_32_llm_obj = BuildLLm(model_url=args.model_url,
                                   model_name=args.model_name,
                                   api_key=args.api_key,
                                   temperature=0.1,
                                   extra_body={"guided_regex": english_only_regex})

    qwen_llm = qwen_awq_32_llm_obj.build_llm()

    start_index = args.start_index  # 0
    end_index = args.end_index  # 10
    chats_dir = args.chats_dir  # chats/10M"
    chat_size = args.chat_size  # 10M

    start = time.perf_counter()

    procs = []
    for process_num in range(start_index, end_index):
        if args.generation_stage == "plan":
            function = run_plans_generation
            p = mp.Process(
                target=function,
                args=(chats_dir, chat_size, process_num)
            )
        elif args.generation_stage == "question":
            function = run_question_generation
            p = mp.Process(
                target=function,
                args=(chats_dir, chat_size, process_num)
            )
        elif args.generation_stage == "answer":
            function = run_answer_generation
            p = mp.Process(
                target=function,
                args=(chats_dir, chat_size, process_num, qwen_llm)
            )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    end = time.perf_counter()

    print(f"Execution time: {end - start:.4f} seconds")
