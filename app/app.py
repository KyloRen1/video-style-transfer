import uvicorn
from fastapi import FastAPI, UploadFile, File

app = FastAPI()


@app.post("/uploadvideo/")
async def create_upload_file(file: UploadFile = File(...)):
    pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)