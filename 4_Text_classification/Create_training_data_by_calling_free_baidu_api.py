import json, requests


def get_api_prediction(text, token):
    """Call sentimental analysis api"""
    url = 'https://aip.baidubce.com/rpc/2.0/nlp/v1/sentiment_classify?charset=UTF-8&access_token='+token
    headers = {'Content-Type': 'application/json'}
    input_text = {'text': text}
    # the input is json format
    input_text = json.dumps(input_text)
    r = requests.post(url, data=input_text, headers=headers)
    return r.json()


if __name__ == '__main__':
    # replace your access token here
    access_token = '24.9818xxxxxxxxxxxx'
    res_dict = get_api_prediction("百度是一家伟大的公司", access_token)
    # if res_dict has key items
    if 'items' in res_dict:
        items = res_dict['items'][0]
        print(items)
        # confidence, range from 0 to 1
        confidence = items['confidence']
        # 0: negative, 1: neutral, 2: positive
        sentiment = items['sentiment']
        # positive probability, range from 0 to 1
        positive_prob = items['positive_prob']
        # negative probability, range from 0 to 1
        negative_prob = items['negative_prob']
        print('sentiment: {}, confidence: {}, positive_prob: {}, negative_prob:{} '.format(sentiment, confidence, positive_prob, negative_prob))
