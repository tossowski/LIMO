

system_prompt = "A conversation between User and Assistant. The user asks a question about an image, and the assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The answer is enclosed within \\box{} tags, i.e. \\box{answer here}\n\n"

few_shot_prompt = ""

question_format = """You should answer a question about an image. You should answer with just one of the options within \\boxed{{}} (For example, if the question is \n'Is the earth flat?\n A: Yes \nB: No', you should answer with \\boxed{{B}}). Here is the question about the image: {question}\n\n"""