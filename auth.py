"""Verifica dei token Supabase per proteggere le route dell'API.

Non validiamo il JWT localmente: deleghiamo la verifica a Supabase stesso
(endpoint /auth/v1/user), così non serve gestire chiavi/segreti di firma
né la loro rotazione lato backend.
"""

import os

import httpx
from fastapi import Header, HTTPException

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")


async def get_current_user(authorization: str | None = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Autenticazione richiesta.")

    token = authorization.removeprefix("Bearer ").strip()

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={"Authorization": f"Bearer {token}", "apikey": SUPABASE_ANON_KEY},
        )

    if resp.status_code != 200:
        raise HTTPException(401, "Sessione non valida o scaduta.")

    return resp.json()
