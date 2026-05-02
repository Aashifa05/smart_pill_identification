from medical_advisor import MedicalAdvisor


def main():
    advisor = MedicalAdvisor()

    pill_info = {
        "name": "ExampleMed 50mg",
        "dosage": "50 mg",
        "usage": "Used for symptomatic relief of example condition.",
        "side_effects": ["nausea", "headache"],
        "precautions": "Avoid with severe liver impairment.",
        "consumption_time": "With food or as directed by label.",
        "confidence": 0.92,
    }

    questions = [
        "What is this pill used for?",
        "What are the side effects?",
        "Is it safe in pregnancy?",
        "Can I increase the dose?",
        "When should I take it?",
        "Identify the pill",
        "Is this medication okay for my elderly parent with kidney issues?",
    ]

    for q in questions:
        print("Q:", q)
        print("A:", advisor.answer_question(pill_info, q))
        print("---")


if __name__ == "__main__":
    main()
