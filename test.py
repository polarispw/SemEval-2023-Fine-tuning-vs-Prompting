td = {"CLS": [{"Variable": ["Premise", "Conclusion", "Stance"],
               "Template": "Human Values are indicated in the premise: {Premise} and the conclusion: {Conclusion} {Stance} it."
               },
              {"第二个模板"}
              ],

      "MBC": [],
      "BCA": [],
      "OA": [],
      "CoT": [],
      }


def wrap_with_template(example, Variable, Template):
    prompt_text = Template
    for v in Variable:
        target = "{" + v + "}"
        prompt_text = prompt_text.replace(target, example[v])
    example["Prompt"] = prompt_text
    return example


example = {"Premise": "@@@",
           "Conclusion": "###",
           "Stance": "!!!",
           "Prompt": ""
           }

input_text = wrap_with_template(example, td["CLS"][0]["Variable"], td["CLS"][0]["Template"])
print(input_text)
