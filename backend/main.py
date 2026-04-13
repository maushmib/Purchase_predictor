from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import random

from schemas import VisitorFeatures
from models import registry

app = FastAPI(
    title="Purchase Predictor API",
    description="Live inference for VIP Merchant Command Center"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def load():
    registry.load_engines()

@app.post("/predict_live")
def predict_live(visitor: VisitorFeatures):
    result = registry.predict(visitor.model_dump())
    return result

# Active websocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

# Simulate live visitors logic
visitors = {}
for i in range(10):
    visitors[f"V-{random.randint(1000, 9999)}"] = {
        "Administrative": random.randint(0, 3),
        "Informational": random.randint(0, 2),
        "ProductRelated": random.randint(1, 15),
        "BounceRates": random.uniform(0, 0.05) if random.random() > 0.3 else random.uniform(0.1, 0.2),
        "PageValues": random.choice([0.0, random.uniform(5, 20), random.uniform(40, 80)]),
        "ProductRelated_Duration": random.uniform(10.0, 1500.0),
        "ExitRates": random.uniform(0.01, 0.1),
        "Month": random.choice(["Nov", "May", "Mar", "Dec"]),
        "Weekend": random.choice([True, False]),
        "VisitorType": random.choice(["Returning_Visitor", "New_Visitor"]),
        "add_to_cart_count": random.randint(0, 2),
        "current_page": "/product/id-" + str(random.randint(100, 999))
    }

async def visitor_simulation_loop():
    while True:
        await asyncio.sleep(random.uniform(4.0, 6.0))
        
        # Pick a random visitor to update
        vid = random.choice(list(visitors.keys()))
        v = visitors[vid]
        
        # Randomly increment some stats
        if random.random() > 0.5:
            v["ProductRelated"] += 1
            v["ProductRelated_Duration"] += random.uniform(10, 60)
            if random.random() > 0.7:
                v["add_to_cart_count"] += 1
                v["PageValues"] += random.uniform(10, 30)
                v["BounceRates"] *= 0.5 # lower bounce rate
            
        # Compile full feature set
        feats = VisitorFeatures(
            Administrative=v["Administrative"],
            Informational=v["Informational"],
            ProductRelated=v["ProductRelated"],
            ProductRelated_Duration=v["ProductRelated_Duration"],
            ExitRates=v["ExitRates"],
            Month=v["Month"],
            Weekend=v["Weekend"],
            VisitorType=v["VisitorType"],
            BounceRates=v["BounceRates"],
            PageValues=v["PageValues"]
        )
        
        # Run inference
        result = registry.predict(feats.model_dump())
        
        # Build payload
        payload = {
            "visitor_id": vid,
            "current_page": v["current_page"],
            "add_to_cart_count": v["add_to_cart_count"],
            "features": feats.model_dump(),
            "prediction": result
        }
        
        await manager.broadcast(payload)


@app.websocket("/ws/live_visitors")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection open
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Start background task when app starts
@app.on_event("startup")
async def start_sim():
    asyncio.create_task(visitor_simulation_loop())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
