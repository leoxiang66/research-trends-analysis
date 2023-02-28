import json
import requests
from typing import Union,List
import aiohttp
from asyncio import run

class InferenceHF:
    headers = {"Authorization": f"Bearer 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"}
    API_URL = "https://api-inference.huggingface.co/models/"

    @classmethod
    def inference(cls, inputs: Union[List[str], str], model_name:str) ->dict:
        payload = dict(
            inputs = inputs,
            options = dict(
                wait_for_model=True
            )
        )

        data = json.dumps(payload)
        response = requests.request("POST", cls.API_URL+model_name, headers=cls.headers, data=data)
        return json.loads(response.content.decode("utf-8"))

    @classmethod
    async def async_inference(cls, inputs: Union[List[str], str], model_name: str) -> dict:
        payload = dict(
            inputs=inputs,
            options=dict(
                wait_for_model=True
            )
        )

        data = json.dumps(payload)

        async with aiohttp.ClientSession() as session:
            async with session.post(cls.API_URL + model_name, data=data, headers=cls.headers) as response:
                return await response.json()


if __name__ == '__main__':
    print(InferenceHF.inference(
        inputs='hi how are you?',
        model_name= 't5-small'
    ))

    print(
        run(InferenceHF.async_inference(
        inputs='hi how are you?',
        model_name='t5-small'
    ))
    )


