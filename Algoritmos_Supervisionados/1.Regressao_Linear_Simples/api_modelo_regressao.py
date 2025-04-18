from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import jobilib

# criar uma instancia do fastapi
app = FastAPI()

# criar uma classe que tera os dados do request body para a API
class request_body(BaseModel):
    horas_estudo: float

# carregar modelo para realizar a predicao
modelo_pontuacao = jobilib.load("modelo_pontuacao.pkl")

@app.post('/predict')

def predict(data: request_body):
    # preparar os dados para a predicao
    input_feature = [[data.horas_estudo]]

    # realizar a predicao
    y_pred = modelo_pontuacao.predict(input_feature)[0].astype(int)

    return {'pontuacao_teste': y_pred.tolist()}