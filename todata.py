import json

# Input data in the format given
data = [
    "<s>[INST]Answer this question truthfully: Can you provide an overview of the lung's squamous cell carcinoma?[/INST] Squamous cell carcinoma of the lung may be classified according to the WHO histological classification system into 4 main types: papillary, clear cell, small cell, and basaloid.</s>",
    "<s>[INST]Answer this question truthfully: What does \"Clear: cell\" mean?[/INST] Clear cell tumors are part of the surface epithelial-stromal tumor group of Ovarian cancers, accounting for 6% of these neoplastic cases. Clear cell tumors are also associated with the pancreas and salivary glands. Benign and borderline variants of this neoplasm are rare, and most cases are malignant. Typically, they are cystic neoplasms with polypoid masses that protrude into the cyst. On microscopic pathological examination, they are composed of cells with clear cytoplasm (that contains glycogen) and hob nail cells (from which the glycogen has been secreted). The pattern may be glandular, papillary or solid.</s>",
    "<s>[INST]Answer this question truthfully: Can you provide me with information regarding statins?[/INST] Two Japanese scientists commenced research into inhibitors of HMG-CoA reductase in 1971 reasoning that organisms might produce such products as the enzyme is important in some essential cell wall components. This work lead to the identification of the first clinically useful compound lovastatin(mevinolin)  from a mould in the mid 1970's. This agent was first used in the more severe forms of hypercholesteraemia in the 1980s followed by landmark trials  with simvastatin that showed the potential for cardio-prevention. Cerivastatin was withdrawn in 2001 because of a ten times higher incidence of rhabdomyolysis than the other statins.</s>",
    "<s>[INST]Answer this question truthfully: What are the historical background and symptoms of Candida-induced vulvovaginitis?[/INST] Symptoms of vulvovaginitis caused by Candida species are indistinguishable and include the following: Pruritus is the most significant symptom Change in the amount and the color of vaginal discharge: It is characterized by a thick, white \"cottage cheese-like\" vaginal discharge Pain on urination (dysuria) Pain on sexual intercourse (dyspareunia) Vulvovaginal soreness Symptoms aggravate a week before the menses</s>"
    
]

# Initialize the output format
intents = []

# Process each entry in the input data
for entry in data:
    # Extract the question and the response using regex
    import re
    match = re.match(r"<s>\[INST\](.*?)\[/INST\](.*?)</s>", entry.strip(), re.DOTALL)
    
    if match:
        question = match.group(1).strip().replace("[INST]", "").replace("[/INST]", "").strip()
        response = match.group(2).strip()
        
        # Clean up the question to make it the tag
        tag = question.split(":")[-1].strip()
        
        # Add the entry to intents
        intents.append({
            "tag": tag,
            "patterns": [question],  # Use the question as a pattern
            "responses": [response]   # The response from the original text
        })

# Prepare the final JSON structure
output = {
    "intents": intents
}

# Convert to JSON format and print or save to file
json_output = json.dumps(output, indent=4)

# Display the result
print(json_output)

# Optionally save to a file
with open("tents.json", "w") as f:
    f.write(json_output)
