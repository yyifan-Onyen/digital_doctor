from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Iterable, Mapping

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(repo_root))
    from digital_doctor.prompt import (  # type: ignore
        DEFAULT_SYSTEM,
        FINAL_DRAFT_PROMPT,
        FINAL_POLISH_PROMPT,
    )
    from digital_doctor.core.text_utils import extract_final  # type: ignore
    from digital_doctor.services.openai_client import call_model  # type: ignore
else:
    from ..core.text_utils import extract_final
    from ..prompt import DEFAULT_SYSTEM, FINAL_DRAFT_PROMPT, FINAL_POLISH_PROMPT
    from ..services.openai_client import call_model


def run_baseline(qa_items: Iterable[Mapping[str, str]], csv_path: str) -> None:
    qa_items = list(qa_items)
    if not qa_items:
        print("No QA items provided; skipping generation.")
        return

    rows = []
    for qa in qa_items:
        q = qa["query"]
        gold_answer = qa.get("gold_answer", "")

        draft_prompt = FINAL_DRAFT_PROMPT.format(
            system_msg=DEFAULT_SYSTEM,
            history_block="",
            refs_block="",
            helper_block="",
            user_msg=q,
        )
        draft_raw = call_model(draft_prompt, json_mode=False)
        draft_text = extract_final(draft_raw)

        polish_prompt = FINAL_POLISH_PROMPT.format(
            system_msg=DEFAULT_SYSTEM,
            refs_block="",
            draft_text=draft_text,
        )
        final_raw = call_model(polish_prompt, json_mode=False)
        final_answer = extract_final(final_raw)

        rows.append(
            {
                "query": q,
                "gold_doctor_answer": gold_answer,
                "draft_prompt": draft_prompt,
                "draft_output": draft_text,
                "final_prompt": polish_prompt,
                "final_output": final_answer,
            }
        )

    fieldnames = [
        "query",
        "gold_doctor_answer",
        "draft_prompt",
        "draft_output",
        "final_prompt",
        "final_output",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    OUTPUT_CSV = "baseline_outputs.csv"

    QA_LIST = [
        {
            "query": "So would you say it's really your OCD and it's not depression?",
            "gold_answer": "One of the first things that you really need to know and understand is that everybody has obsessive thoughts. Your partner has obsessive thoughts, and I have obsessive thoughts. You may have a higher frequency of obsessive thoughts, but what's different is your response to them. You got hooked. An alarm happened when you thought about jumping on the tracks or jumping off the cruise ship, where the rest of us would think, \"Oh, that's ridiculous. I must be tired.\" And then the anxiety starts rolling. It's not so much for you never to have obsessive thoughts again. It's more to change how you respond to them.",
        },
        {
            "query": "I don't really feel depressed, but I did check that item about suicidal thoughts. How should I think about that?",
            "gold_answer": "You don't show up as depressed on that self-report inventory that you filled out. The thoughts are not with true intent. It's something that's uncomfortable for you. It's more of an obsession than true suicidal ideation. The goal here is for you to develop a tolerance for that happening and not engage in any behaviors to artificially reduce it, because then you're chasing your tail.",
        },
        {
            "query": "I feel like this is all very new to me, even finding the vocabulary to talk about it.",
            "gold_answer": "That makes a lot of sense. Often when people first start to put language around their inner experience, it feels strange. Being able to label something as an obsession or a compulsion is actually very important, because once you label it, you can see that it's a mental construction and not reality itself.",
        },
        {
            "query": "When I say harming myself, I meant things like jumping off high places or off a cruise ship.",
            "gold_answer": "So when it's a fear of jumping off high places, the important thing for us to understand is whether it shows up as an image, a thought, or an impulse. The anxiety usually comes from the obsessional image or thought, and then what maintains it is how you respond. Our work is to understand those triggers and then change the response.",
        },
        {
            "query": "Sometimes it feels like an image, and sometimes just a thought.",
            "gold_answer": "Right. And often if you entertain the thought longer, it can turn into an image. The whole thing can ratchet up. That's why we're interested in understanding what happens when the thought first appears, what you do next, and how that response either calms things temporarily or keeps the cycle going.",
        },
        {
            "query": "A lot of these thoughts come up when I'm commuting or doing the same routine every day.",
            "gold_answer": "That's a classic example of conditioned triggers. When something happens at the same time and place every day, your brain starts pairing it with the obsession. When you're distracted, those thoughts get crowded out. But you can't be busy all the time, so the work is learning to respond differently when your mind is more fertile ground for obsessions.",
        },
        {
            "query": "When the thoughts were worse, I tried calming myself, thinking positive thoughts, breathing, or even calling my own bluff.",
            "gold_answer": "Those are all attempts to reduce anxiety, and they make perfect sense. The problem is that they all serve the same function: they tell your brain that the thought is dangerous and needs to be managed. That's how the loop gets maintained, even though your intention was to help yourself.",
        },
        {
            "query": "I don't feel suicidal, but I have suicidal thoughts. Explaining that feels strange.",
            "gold_answer": "That's actually a very common OCD symptom. People have thoughts or images of harming themselves without any true intent. What makes it OCD is not the content of the thought, but the alarm and the response. The treatment is not about making the thoughts disappear, but about changing your relationship to them.",
        },
        {
            "query": "Sometimes I even feel an impulse, not just a thought.",
            "gold_answer": "Some people do report an impulse. Others report images. The key thing is that the impulse itself doesn't mean anything is going to happen. It's part of the obsessional experience. What matters is whether you treat it as a signal that requires action or as a mental event that can rise and fall on its own.",
        },
        {
            "query": "I used to have violent or horrific images too, like when passing places where someone had died.",
            "gold_answer": "That fits very well with how obsessions work. Sometimes it's being the perpetrator, sometimes it's observing violence. The brain pulls you toward the image, almost like rubber-necking. What we look at is whether you do anything to reduce discomfort or whether the image just comes and goes on its own.",
        },
        {
            "query": "I don't think I really did compulsions with those images. Sometimes I just moved on.",
            "gold_answer": "That's important to notice. When you don't engage in rituals, the image often fades. That gives us information about what happens when the obsession isn't fed. Not every intrusive image turns into OCD; it's the response that determines that.",
        },
        {
            "query": "I used to get anxious about things like the number 13, but it didn't last very long.",
            "gold_answer": "That's a good example of how OCD symptoms can wax and wane or change form. Sometimes a theme fades because you stop responding to it, even without realizing it. Other times the content shifts to something else, but the underlying process is the same.",
        },
        {
            "query": "I sometimes feel like I'm supposed to be anxious, almost reminding myself.",
            "gold_answer": "That belief is very powerful. If you tell yourself that being anxious is what keeps bad things from happening, your autonomic nervous system stays activated. Your blood pressure goes up, your breathing changes, and the whole cycle keeps reinforcing itself.",
        },
        {
            "query": "I don't actually believe that deep down, but it still feels automatic.",
            "gold_answer": "Right, and that's why we work on both thinking patterns and behaviors. Even if you don't consciously endorse the belief, responding as if it's true keeps the system going. Part of treatment is noticing that mismatch and practicing a different response.",
        },
    ]

    run_baseline(QA_LIST, OUTPUT_CSV)
