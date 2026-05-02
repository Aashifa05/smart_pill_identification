#!/usr/bin/env python
"""
Script to populate PillInfo database with all 10 pill types
Run with: python populate_pills.py
"""
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Detection_and_Analysis_of_Pill.settings')
django.setup()

from Users.models import PillInfo

pills_data = [
    {
        "name": "Alaxan",
        "usage": "For relief of mild to moderate pain, headaches, muscle pain, and menstrual cramps",
        "dosage": "500mg tablet - Take 1 tablet every 4-6 hours as needed, do not exceed 8 tablets per day",
        "consumption_time": "Anytime",
        "food_relation": "Can be taken with or without food",
        "side_effects": "Dizziness, nausea, stomach upset, allergic reactions",
        "precautions": "Not for use in children under 12 years. Avoid if allergic to ibuprofen."
    },
    {
        "name": "Bactidol",
        "usage": "Sore throat lozenges with antiseptic properties. Used for throat pain and mild throat infections",
        "dosage": "1 lozenge - Suck 1 lozenge every 2 hours as needed, maximum 8 lozenges per day",
        "consumption_time": "Anytime during the day",
        "food_relation": "Can be used anytime",
        "side_effects": "Local irritation, allergic reactions in sensitive individuals",
        "precautions": "For adults and children above 5 years. Discontinue if severe throat pain persists."
    },
    {
        "name": "Bioflu",
        "usage": "Relieves symptoms of common cold, flu, fever, body aches, and headaches",
        "dosage": "500mg capsule - Take 1 capsule every 4-6 hours as needed, maximum 6 capsules per day",
        "consumption_time": "Anytime",
        "food_relation": "Take with food if stomach upset occurs",
        "side_effects": "Dizziness, drowsiness, nausea, allergic reactions",
        "precautions": "Not for children under 12 years. May cause drowsiness, avoid driving."
    },
    {
        "name": "Biogesic",
        "usage": "For relief of mild to moderate pain and fever, including headaches and muscle pain",
        "dosage": "500mg tablet - Take 1 tablet every 4-6 hours, not to exceed 4000mg per day",
        "consumption_time": "Anytime",
        "food_relation": "Can be taken with or without food",
        "side_effects": "Liver damage (rare with overdose), allergic reactions, rash",
        "precautions": "Do not exceed recommended dose. Not for children under 2 years."
    },
    {
        "name": "DayZinc",
        "usage": "Zinc supplement for immune system support and cold symptom relief",
        "dosage": "15mg tablet - Take 1 tablet daily with water, preferably with meals",
        "consumption_time": "Morning or with meals",
        "food_relation": "Take with food to enhance absorption",
        "side_effects": "Nausea, bad taste in mouth, copper deficiency with long-term use",
        "precautions": "Do not exceed 40mg per day. Not recommended for prolonged use without medical advice."
    },
    {
        "name": "Decolgen",
        "usage": "For relief of nasal congestion, cough, and cold symptoms",
        "dosage": "2 tablets - Take 2 tablets every 4-6 hours as needed, maximum 8 tablets per day",
        "consumption_time": "Anytime",
        "food_relation": "Can be taken with or without food",
        "side_effects": "Drowsiness, dizziness, dry mouth, increased heart rate",
        "precautions": "May cause drowsiness. Not suitable for children under 12 years."
    },
    {
        "name": "Fish Oil",
        "usage": "Omega-3 supplement for heart health, joint support, and brain function",
        "dosage": "1000mg softgel - Take 1-2 softgels daily with meals",
        "consumption_time": "With meals (breakfast or dinner)",
        "food_relation": "Take with food to reduce fishy aftertaste and improve absorption",
        "side_effects": "Fishy aftertaste, mild gastrointestinal upset, bleeding (rare with high doses)",
        "precautions": "May increase bleeding risk if taking anticoagulants. Consult doctor if pregnant."
    },
    {
        "name": "Kremil S",
        "usage": "Antacid and anti-gas medication for relief of heartburn, acid indigestion, and bloating",
        "dosage": "2-4 tablets - Take 2-4 tablets after meals or as needed, maximum 12 tablets per day",
        "consumption_time": "After meals or as needed",
        "food_relation": "Take after meals",
        "side_effects": "Constipation, nausea, aluminum-related side effects with prolonged use",
        "precautions": "Not for regular use. Consult doctor if symptoms persist beyond 2 weeks."
    },
    {
        "name": "Medicol",
        "usage": "Ibuprofen-based pain reliever for fever, headaches, muscle pain, and menstrual cramps",
        "dosage": "200mg tablet - Take 1-2 tablets every 4-6 hours as needed, not to exceed 6 tablets per day",
        "consumption_time": "Anytime",
        "food_relation": "Take with food or milk to reduce stomach irritation",
        "side_effects": "Stomach upset, nausea, dizziness, allergic reactions",
        "precautions": "Not for children under 12 years. Avoid if allergic to NSAIDs."
    },
    {
        "name": "Neozep",
        "usage": "Cold and cough reliever with decongestant for congestion, cough, and sneezing",
        "dosage": "2 capsules - Take 2 capsules every 4-6 hours as needed, maximum 6 capsules per day",
        "consumption_time": "Anytime",
        "food_relation": "Can be taken with or without food",
        "side_effects": "Drowsiness, dizziness, increased heart rate, dry mouth",
        "precautions": "May cause drowsiness. Not for children under 12 years. Avoid driving."
    }
]

# Add or update pills in the database
for pill in pills_data:
    pill_obj, created = PillInfo.objects.update_or_create(
        name=pill['name'],
        defaults={
            'usage': pill['usage'],
            'dosage': pill['dosage'],
            'consumption_time': pill['consumption_time'],
            'food_relation': pill['food_relation'],
            'side_effects': pill['side_effects'],
            'precautions': pill['precautions']
        }
    )
    status = "Created" if created else "Updated"
    print(f"{status}: {pill['name']}")

print("\nAll pills have been populated successfully!")
