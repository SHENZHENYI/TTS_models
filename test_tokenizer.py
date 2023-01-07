from src.utils.text import text_to_sequence

texts = [
    'tokeer .',
    ' eee'
]
for t in texts:
    res =text_to_sequence(t, ['english_cleaners'])
    print(t, res)