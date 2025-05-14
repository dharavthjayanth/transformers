## How to run the project 

### Frontend
- nvm use 18
- npm run dev

### Backend
- conda create --name venv python=3.10 (create a local venv)
- conda activate venv
- pip install -r requirement.txt
- uvicorn main:app --reload