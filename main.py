import intel_extension_for_pytorch as ipex
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from routers.image import router as image_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://local.wamellow.com",
        "https://wamellow.com"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# set VS2022INSTALLDIR=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools
# call "C:\Program Files (x86)\Intel\oneAPI\compiler\latest\env\vars.bat"
# call "C:\Program Files (x86)\Intel\oneAPI\mkl\latest\env\vars.bat"


@app.middleware("http")
async def add_process_time_header(request: Request, next):

    if request.method == "OPTIONS":
        return Response(status_code=204)

    response = await next(request)
    return response

app.include_router(image_router)

# fastapi dev main.py


@app.get("/")
def read_root():
    return {"gpu": ipex.xpu.get_device_name(0)}


app.mount("/static", StaticFiles(directory="output"), name="output")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="100.65.0.2", port=8000)
