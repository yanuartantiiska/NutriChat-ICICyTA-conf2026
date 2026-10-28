"""
Synthetic deictic query templates for NutriChat Exp04.

IMPORTANT
---------
These utterances are synthetic multimodal scenario queries.
They are NOT real patient utterances.

The templates deliberately do NOT expose:
- ingredient identity,
- ground-truth nutrient values,
- laboratory values,
- clinical thresholds,
- recommendation labels.

Food identity is expected to come from the visual modality.
Clinical context is supplied separately by the clinical encoder.
"""


# =========================================================
# Condition expressions
# =========================================================

CONDITION_EXPRESSIONS = {

    "hypertension": [
        "untuk kondisi hipertensi saya",
        "jika saya sedang mengontrol tekanan darah",
        "untuk membantu menjaga tekanan darah saya",
        "dengan kondisi tekanan darah tinggi saya",
    ],

    "dyslipidemia": [
        "untuk kondisi dislipidemia saya",
        "jika saya sedang mengontrol kolesterol",
        "untuk membantu menjaga kadar lemak darah saya",
        "dengan kondisi kolesterol tinggi saya",
    ],

    "both": [
        "untuk kondisi hipertensi dan dislipidemia saya",
        "jika saya perlu menjaga tekanan darah dan kolesterol",
        "untuk membantu mengontrol tekanan darah dan kadar lemak darah saya",
        "dengan kondisi tekanan darah dan kolesterol yang perlu saya jaga",
    ],

    "neither": [
        "untuk pola makan saya",
        "untuk kebutuhan makan saya",
        "dalam pola makan sehari-hari saya",
        "untuk pilihan makan saya",
    ],
}


# =========================================================
# Templates
#
# {food_ref}      = deictic reference
# {condition_ref} = optional condition phrase
#
# Each family represents a different surface construction.
# =========================================================

TEMPLATES = {

    # -----------------------------------------------------
    # FOOD SUITABILITY
    # -----------------------------------------------------

    "food_suitability": {

        "MM_FS01": [
            "Apakah {food_ref} sesuai {condition_ref}?",
            "Apakah {food_ref} cocok {condition_ref}?",
        ],

        "MM_FS02": [
            "Bagaimana kesesuaian {food_ref} {condition_ref}?",
            "Bagaimana menurut Anda kesesuaian {food_ref} {condition_ref}?",
        ],

        "MM_FS03": [
            "Bolehkah saya memilih {food_ref} {condition_ref}?",
            "Apakah saya dapat memilih {food_ref} {condition_ref}?",
        ],

        "MM_FS04": [
            "Saya ingin tahu apakah {food_ref} merupakan pilihan yang sesuai {condition_ref}.",
            "Saya ingin memastikan apakah {food_ref} dapat menjadi pilihan {condition_ref}.",
        ],

        "MM_FS05": [
            "Menurut informasi pada foto, apakah {food_ref} sesuai {condition_ref}?",
            "Dari makanan yang terlihat, apakah {food_ref} cocok {condition_ref}?",
        ],
    },


    # -----------------------------------------------------
    # NUTRITION CONCERN
    # -----------------------------------------------------

    "nutrition_concern": {

        "MM_NC01": [
            "Apa yang perlu saya perhatikan dari kandungan {food_ref} {condition_ref}?",
            "Hal gizi apa yang perlu saya perhatikan dari {food_ref} {condition_ref}?",
        ],

        "MM_NC02": [
            "Bagaimana saya sebaiknya menilai kandungan gizi {food_ref} {condition_ref}?",
            "Apa yang penting diperhatikan dari sisi gizi pada {food_ref} {condition_ref}?",
        ],

        "MM_NC03": [
            "Apakah ada kandungan pada {food_ref} yang perlu saya perhatikan {condition_ref}?",
            "Adakah aspek gizi dari {food_ref} yang perlu diperhatikan {condition_ref}?",
        ],

        "MM_NC04": [
            "Saya ingin memahami kandungan gizi {food_ref} {condition_ref}.",
            "Saya ingin mengetahui hal penting tentang kandungan {food_ref} {condition_ref}.",
        ],

        "MM_NC05": [
            "Dari sisi nutrisi, apa yang perlu diperhatikan pada {food_ref} {condition_ref}?",
            "Jika dilihat dari nutrisinya, apa yang perlu saya perhatikan dari {food_ref} {condition_ref}?",
        ],
    },


    # -----------------------------------------------------
    # PORTION INQUIRY
    # -----------------------------------------------------

    "portion_inquiry": {

        "MM_PI01": [
            "Bagaimana dengan porsi {food_ref} {condition_ref}?",
            "Apakah porsi {food_ref} perlu saya perhatikan {condition_ref}?",
        ],

        "MM_PI02": [
            "Apakah jumlah {food_ref} perlu disesuaikan {condition_ref}?",
            "Bagaimana saya sebaiknya mempertimbangkan jumlah {food_ref} {condition_ref}?",
        ],

        "MM_PI03": [
            "Apakah porsi {food_ref} terlihat sesuai {condition_ref}?",
            "Bagaimana kesesuaian porsi {food_ref} {condition_ref}?",
        ],

        "MM_PI04": [
            "Saya ingin tahu apakah porsi {food_ref} sudah sesuai {condition_ref}.",
            "Saya ingin memastikan porsi {food_ref} dapat dipertimbangkan {condition_ref}.",
        ],

        "MM_PI05": [
            "Untuk sekali makan, bagaimana sebaiknya saya mempertimbangkan porsi {food_ref} {condition_ref}?",
            "Untuk satu kali makan, apakah porsi {food_ref} perlu disesuaikan {condition_ref}?",
        ],
    },


    # -----------------------------------------------------
    # FOOD GUIDANCE
    # -----------------------------------------------------

    "food_guidance": {

        "MM_FG01": [
            "Apa yang perlu saya pertimbangkan sebelum mengonsumsi {food_ref} {condition_ref}?",
            "Apa yang sebaiknya saya perhatikan sebelum memilih {food_ref} {condition_ref}?",
        ],

        "MM_FG02": [
            "Bagaimana sebaiknya saya menyikapi {food_ref} {condition_ref}?",
            "Bagaimana sebaiknya saya mempertimbangkan {food_ref} {condition_ref}?",
        ],

        "MM_FG03": [
            "Apa saran umum terkait {food_ref} {condition_ref}?",
            "Apa panduan umum untuk mempertimbangkan {food_ref} {condition_ref}?",
        ],

        "MM_FG04": [
            "Saya ingin mendapatkan panduan mengenai {food_ref} {condition_ref}.",
            "Saya membutuhkan arahan untuk mempertimbangkan {food_ref} {condition_ref}.",
        ],

        "MM_FG05": [
            "Jika saya ingin memilih {food_ref}, apa yang perlu diperhatikan {condition_ref}?",
            "Sebelum memutuskan memilih {food_ref}, apa yang perlu saya pertimbangkan {condition_ref}?",
        ],
    },
}


# =========================================================
# Deictic food references
# =========================================================

FOOD_REFERENCES = [
    "makanan ini",
    "hidangan ini",
    "makanan pada foto ini",
    "hidangan pada foto ini",
    "makanan yang saya unggah",
    "hidangan yang saya unggah",
]
# =========================================================
# Natural conversational prefixes
# =========================================================

QUERY_PREFIXES = [
    "",
    "Mohon bantu saya, ",
    "Boleh saya bertanya, ",
    "Tolong bantu saya, ",
    "Mohon informasinya, ",
    "Untuk makanan ini, ",
    "Mohon penjelasannya, ",
    "Bisa bantu saya, ",
]

# =========================================================
# Basic structural validation
# =========================================================

EXPECTED_QUERY_TYPES = {
    "food_suitability",
    "nutrition_concern",
    "portion_inquiry",
    "food_guidance",
}


def validate_templates():

    assert set(TEMPLATES.keys()) == EXPECTED_QUERY_TYPES

    family_ids = []

    template_count = 0

    for query_type, families in TEMPLATES.items():

        assert len(families) == 5

        for family_id, templates in families.items():

            family_ids.append(family_id)

            assert len(templates) >= 2

            for template in templates:

                assert "{food_ref}" in template

                assert "{condition_ref}" in template

                template_count += 1

    assert len(family_ids) == 20
    assert len(set(family_ids)) == 20

    assert len(FOOD_REFERENCES) == 6

    assert len(QUERY_PREFIXES) == 8
    assert len(set(QUERY_PREFIXES)) == 8

    return {
        "query_types": len(TEMPLATES),
        "families": len(family_ids),
        "surface_templates": template_count,
        "food_references": len(FOOD_REFERENCES),
        "query_prefixes": len(QUERY_PREFIXES),
    }


if __name__ == "__main__":

    result = validate_templates()

    print("=" * 72)
    print("EXP04 MULTIMODAL TEXT TEMPLATE AUDIT")
    print("=" * 72)

    for key, value in result.items():
        print(f"{key:20s}: {value}")

    print(
        "\nEXP04 MULTIMODAL TEXT TEMPLATES: PASS"
    )