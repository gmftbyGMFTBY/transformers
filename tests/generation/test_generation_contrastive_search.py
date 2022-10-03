import torch

from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OPTForCausalLM,
    T5ForConditionalGeneration,
)


# make sure the scripts runs on GPU devices
def move_data_to_cuda(data):
    new_data = {}
    for key, value in data.items():
        try:
            value = value.cuda()
            new_data[key] = value
        except:
            new_data[key] = value
    return new_data


# test opt model
print("-" * 25, "test OPT-350M model", "-" * 25)
tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-350m")
# for batch version generation
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
model.cuda()

inputs = tokenizer(["DeepMind Company is", "Microsoft Company is"], return_tensors="pt", padding=True)
inputs = move_data_to_cuda(inputs)

# generation_output = model.generate(**inputs, return_dict_in_generate=True, output_scores=True, penalty_alpha=0.6, top_k=4, stopping_criteria=stopping_criteria, output_hidden_states=True)
generation_output = model.contrastive_search(
    **inputs,
    return_dict_in_generate=True,
    output_scores=True,
    penalty_alpha=0.6,
    top_k=4,
    max_length=256,
    output_hidden_states=True,
)

first_sequences = tokenizer.decode(generation_output[0][0])
second_sequences = tokenizer.decode(generation_output[0][1])
print("OPT-325M Contrastive Search Output-1:\n", "\t", first_sequences, "\n")
print("OPT-325M Contrastive Search Output-2:\n", "\t", second_sequences, "\n")

generation_output = model.generate(**inputs, max_length=256, do_sample=True, top_p=0.92, top_k=0.0)
first_sequences = tokenizer.decode(generation_output[0])
second_sequences = tokenizer.decode(generation_output[1])
print("OPT-325M Nucleus Sampling Output-1:\n", "\t", first_sequences, "\n")
print("OPT-325M Nucleus Sampling Output-2:\n", "\t", second_sequences, "\n")

# test gpt model
print("-" * 25, "test GPT2 Large model", "-" * 25)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2-large")
model.cuda()

inputs = tokenizer(["DeepMind Company is", "Microsoft Company is"], return_tensors="pt", padding=True)
inputs = move_data_to_cuda(inputs)

generation_output = model.contrastive_search(
    **inputs,
    return_dict_in_generate=True,
    output_scores=True,
    penalty_alpha=0.6,
    top_k=4,
    max_length=256,
    output_hidden_states=True,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

first_sequences = tokenizer.decode(generation_output[0][0])
second_sequences = tokenizer.decode(generation_output[0][1])
print("GPT2-Large Contrastive Search Output-1:\n", "\t", first_sequences, "\n")
print("GPT2-Large Contrastive Search Output-2:\n", "\t", second_sequences, "\n")

generation_output = model.generate(**inputs, max_length=256, do_sample=True, top_p=0.92, top_k=0.0)
first_sequences = tokenizer.decode(generation_output[0])
second_sequences = tokenizer.decode(generation_output[1])
print("GPT2-Large Nucleus Sampling Output-1:\n", "\t", first_sequences, "\n")
print("GPT2-Large Nucleus Sampling Output-2:\n", "\t", second_sequences, "\n")


print("-" * 25, "test T5 model", "-" * 25)

ARTICLE = """ New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
2010 marriage license application, according to court documents.
Prosecutors said the marriages were part of an immigration scam.
On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s
Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
"""
ARTICLE = "summarize: " + ARTICLE.strip()

tokenizer = AutoTokenizer.from_pretrained("flax-community/t5-base-cnn-dm")
model = T5ForConditionalGeneration.from_pretrained("flax-community/t5-base-cnn-dm")
model.cuda()

inputs = tokenizer(
    [ARTICLE, ARTICLE[-256:]],
    add_special_tokens=False,
    truncation=True,
    max_length=512,
    return_tensors="pt",
    padding=True,
)
inputs = move_data_to_cuda(inputs)

# donot use model.contrastive_search, since model.generate will prepare the decoder_inptu_ids for encoder-decoder model
generation_output = model.generate(
    **inputs,
    return_dict_in_generate=True,
    output_scores=True,
    penalty_alpha=0.5,
    top_k=5,
    max_length=32,
    output_hidden_states=True,
)

first_sequences = tokenizer.decode(generation_output[0][0])
second_sequences = tokenizer.decode(generation_output[0][1])
print("T5 Contrastive Search Output-1:\n", "\t", first_sequences)
print("T5 Contrastive Search Output-2:\n", "\t", second_sequences)

generation_output = model.generate(**inputs, max_length=32, do_sample=True, top_p=0.92, top_k=0.0)
first_sequences = tokenizer.decode(generation_output[0])
second_sequences = tokenizer.decode(generation_output[1])
print("T5 Nucleus Sampling Output-1:\n", "\t", first_sequences)
print("T5 Nucleus Sampling Output-2:\n", "\t", second_sequences)


print("-" * 25, "test BART model", "-" * 25)

ARTICLE = """ New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
2010 marriage license application, according to court documents.
Prosecutors said the marriages were part of an immigration scam.
On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s
Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
"""
ARTICLE = "summarize: " + ARTICLE.strip()

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
model.cuda()

inputs = tokenizer(
    [ARTICLE, ARTICLE[-256:]],
    add_special_tokens=False,
    truncation=True,
    max_length=512,
    return_tensors="pt",
    padding=True,
)
inputs = move_data_to_cuda(inputs)

# donot use model.contrastive_search, since model.generate will prepare the decoder_inptu_ids for encoder-decoder model
generation_output = model.generate(
    **inputs,
    return_dict_in_generate=True,
    output_scores=True,
    penalty_alpha=0.5,
    top_k=5,
    max_length=64,
    output_hidden_states=True,
)

first_sequences = tokenizer.decode(generation_output[0][0])
second_sequences = tokenizer.decode(generation_output[0][1])
print("BART Contrastive Search Output-1:\n", "\t", first_sequences)
print("BART Contrastive Search Output-2:\n", "\t", second_sequences)

generation_output = model.generate(**inputs, max_length=64, do_sample=True, top_p=0.92, top_k=0.0)
first_sequences = tokenizer.decode(generation_output[0])
second_sequences = tokenizer.decode(generation_output[1])
print("BART Nucleus Sampling Output-1:\n", "\t", first_sequences)
print("BART Nucleus Sampling Output-2:\n", "\t", second_sequences)

# test simctg on wikitext103 model
print("-" * 25, "test SimCTG model on WikiText-103", "-" * 25)
tokenizer = GPT2Tokenizer.from_pretrained("cambridgeltl/simctg_wikitext103")
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("cambridgeltl/simctg_wikitext103")
model.cuda()

inputs = tokenizer(
    "Butt criticized Donald 's controls in certain situations in the game , as well as the difficulty of some levels"
    " and puzzles . Buchanan also criticized the controls , calling",
    return_tensors="pt",
    padding=True,
)
inputs = move_data_to_cuda(inputs)

generation_output = model.generate(
    **inputs,
    return_dict_in_generate=True,
    output_scores=True,
    penalty_alpha=0.6,
    top_k=8,
    max_length=128,
    output_hidden_states=True,
)

first_sequences = tokenizer.decode(generation_output[0][0])
print("SimCTG-wikitext-103 Contrastive Search Output-1:\n", "\t", first_sequences, "\n")

generation_output = model.generate(**inputs, max_length=128, do_sample=True, top_p=0.92, top_k=0.0)
first_sequences = tokenizer.decode(generation_output[0])
print("SimCTG-wikitext-103 Nucleus Sampling Output-1:\n", "\t", first_sequences, "\n")


# test Chinese Open-Domain Dialog model
print("-" * 25, "test Chinese open-domain dialog", "-" * 25)
tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/simctg_lccc_dialogue")
tokenizer.padding_side = "left"
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
model = GPT2LMHeadModel.from_pretrained("cambridgeltl/simctg_lccc_dialogue")
model.cuda()

inputs = tokenizer(
    " [SEP] ".join(["刺猬很可爱！以前别人送了只没养，味儿太大！", "是很可爱但是非常臭", "是啊，没办法养", "那个怎么养哦不会扎手吗"]), return_tensors="pt", padding=True
)
inputs = move_data_to_cuda(inputs)

generation_output = model.generate(
    **inputs,
    return_dict_in_generate=True,
    output_scores=True,
    penalty_alpha=0.6,
    top_k=5,
    max_length=64,
    output_hidden_states=True,
)

first_sequences = tokenizer.decode(generation_output[0][0])
print("SimCTG-LCCC Contrastive Search Output-1:\n", "\t", first_sequences, "\n")

generation_output = model.generate(**inputs, max_length=128, do_sample=True, top_p=0.92, top_k=0.0)
first_sequences = tokenizer.decode(generation_output[0])
print("SimCTG-LCCC Nucleus Sampling Output-1:\n", "\t", first_sequences, "\n")


# test off-the-shelf Chinese Generation model
print("-" * 25, "test off-the-shelf Chinese Generation", "-" * 25)
tokenizer = AutoTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
tokenizer.padding_side = "left"
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
model.cuda()

inpt_1 = torch.LongTensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize("苹果公司")))
inpt_2 = torch.LongTensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize("百节年为首，春节是中华民族最隆重的传统佳节。它不仅集中体现了中华")))
inputs = torch.stack(
    [
        torch.cat(
            [torch.ones(len(inpt_2) - len(inpt_1)).to(torch.long).fill_(tokenizer.pad_token_id), inpt_1], dim=-1
        ),
        inpt_2,
    ]
)
inputs = {"input_ids": inputs.cuda()}

generation_output = model.generate(
    **inputs,
    return_dict_in_generate=True,
    output_scores=True,
    penalty_alpha=0.6,
    top_k=3,
    max_length=128,
    output_hidden_states=True,
    pad_token_id=tokenizer.pad_token_id,
)

first_sequences = tokenizer.decode(generation_output[0][0])
second_sequences = tokenizer.decode(generation_output[0][1])
print("Contrastive Search Output-1:\n", "\t", first_sequences, "\n")
print("Contrastive Search Output-2:\n", "\t", second_sequences, "\n")

generation_output = model.generate(**inputs, max_length=128, do_sample=True, top_p=0.92, top_k=0.0)
first_sequences = tokenizer.decode(generation_output[0])
second_sequences = tokenizer.decode(generation_output[1])
print("Nucleus Sampling Output-1:\n", "\t", first_sequences, "\n")
print("Nucleus Sampling Output-2:\n", "\t", second_sequences, "\n")


# test off-the-shelf japaness generation model
print("-" * 25, "test Japaness generation model", "-" * 25)
tokenizer = AutoTokenizer.from_pretrained("colorfulscoop/gpt2-small-ja")
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("colorfulscoop/gpt2-small-ja")
model.cuda()

inputs = tokenizer("臥龍桜（がりゅうざくら）は、岐阜県高山市一之宮町にある一本桜。龍が地", return_tensors="pt", padding=True)
inputs = move_data_to_cuda(inputs)

generation_output = model.generate(
    **inputs,
    return_dict_in_generate=True,
    output_scores=True,
    penalty_alpha=0.6,
    top_k=5,
    max_length=128,
    output_hidden_states=True,
)

first_sequences = tokenizer.decode(generation_output[0][0])
print("Contrastive Search Output-1:\n", "\t", first_sequences, "\n")

generation_output = model.generate(**inputs, max_length=128, do_sample=True, top_p=0.92, top_k=0.0)
first_sequences = tokenizer.decode(generation_output[0])
print("Nucleus Sampling Output-1:\n", "\t", first_sequences, "\n")


# test off-the-shelf korean generation model
print("-" * 25, "test Korean generation model", "-" * 25)
tokenizer = AutoTokenizer.from_pretrained("skt/ko-gpt-trinity-1.2B-v0.5")
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("skt/ko-gpt-trinity-1.2B-v0.5")
model.cuda()

inputs = tokenizer(r"인간처럼 생각하고, 행동하는 \'지능\'을 통해 인류가 이제까지 풀지 못했던", return_tensors="pt", padding=True)
inputs = move_data_to_cuda(inputs)

generation_output = model.generate(
    **inputs,
    return_dict_in_generate=True,
    output_scores=True,
    penalty_alpha=0.6,
    top_k=5,
    max_length=64,
    output_hidden_states=True,
)

first_sequences = tokenizer.decode(generation_output[0][0])
print("Contrastive Search Output-1:\n", "\t", first_sequences, "\n")

generation_output = model.generate(**inputs, max_length=64, do_sample=True, top_p=0.92, top_k=0.0)
first_sequences = tokenizer.decode(generation_output[0])
print("Nucleus Sampling Output-1:\n", "\t", first_sequences, "\n")
