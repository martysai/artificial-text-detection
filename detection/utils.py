from typing import List

SRC_LANG = 'ru'
TRG_LANG = 'en'


def get_mock_dataset() -> List[dict]:
    dataset = [
        {
            SRC_LANG: 'добрый вечер',
            TRG_LANG: 'good evening',
        },
        {
            SRC_LANG: 'прошу прощения',
            TRG_LANG: 'i am sorry',
        }
    ]
    return dataset
