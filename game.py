import asyncio
import json
import uuid
from dataclasses import dataclass, field
from fastapi import WebSocket

import gemini_service


@dataclass
class Player:
    ws: WebSocket | None
    player_id: int
    image_data: str | None = None
    character: dict | None = None
    ready: bool = False


@dataclass
class Room:
    room_id: str
    players: dict[int, Player] = field(default_factory=dict)
    state: str = "waiting"  # waiting, playing, finished
    _next_id: int = 1


class RoomManager:
    def __init__(self):
        self.rooms: dict[str, Room] = {}
        self._waiting: Player | None = None
        self._waiting_event: asyncio.Event | None = None

    async def handle_connection(self, ws: WebSocket):
        await ws.accept()

        # If someone is already waiting, match immediately
        if self._waiting is not None:
            partner = self._waiting
            waiting_event = self._waiting_event
            self._waiting = None
            self._waiting_event = None

            room_id = uuid.uuid4().hex[:8]
            room = Room(room_id=room_id)
            self.rooms[room_id] = room

            partner.player_id = 1
            room.players[1] = partner

            player = Player(ws=ws, player_id=2)
            room.players[2] = player

            # Signal the waiting player that they've been matched.
            # Their inline waiting loop will exit and they'll start
            # their own _run_player.
            if waiting_event:
                waiting_event.set()

            # Notify both — handle partner disconnect gracefully
            try:
                await partner.ws.send_json({
                    "type": "joined",
                    "player_id": 1,
                    "players_in_room": 2,
                })
            except Exception:
                # Partner's connection is dead, fall back to AI for this player
                room.players.pop(1, None)
                self.rooms.pop(room_id, None)
                await self._start_ai_battle(player, ws)
                return

            await ws.send_json({
                "type": "joined",
                "player_id": 2,
                "players_in_room": 2,
            })
            for p in room.players.values():
                if p.ws is not None:
                    try:
                        await p.ws.send_json({
                            "type": "both_joined",
                            "player_id": p.player_id,
                        })
                    except Exception:
                        pass

            # Only run this player's message loop.
            # The partner's handle_connection will run their own _run_player
            # after their waiting loop exits.
            await self._run_player(room, player)
            return

        # No one waiting — this player waits
        player = Player(ws=ws, player_id=0)
        self._waiting = player
        event = asyncio.Event()
        self._waiting_event = event

        await ws.send_json({
            "type": "waiting",
        })

        # Wait for match, skip, or timeout (30s).
        # Single inline loop — no separate task, no concurrent WebSocket readers.
        loop = asyncio.get_event_loop()
        deadline = loop.time() + 30.0
        skip_requested = False
        disconnected = False

        while not event.is_set():
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                data = await asyncio.wait_for(
                    ws.receive_text(),
                    timeout=min(remaining, 1.0),
                )
                msg = json.loads(data)
                if msg.get("type") == "skip":
                    skip_requested = True
                    break
            except asyncio.TimeoutError:
                continue
            except Exception:
                # Connection lost while waiting
                disconnected = True
                break

        if disconnected:
            if self._waiting is player:
                self._waiting = None
                self._waiting_event = None
            return

        matched_by_opponent = (
            not skip_requested
            and self._waiting is not player
        )

        if not matched_by_opponent:
            # No opponent found or skip — create AI opponent
            if self._waiting is player:
                self._waiting = None
                self._waiting_event = None

            await self._start_ai_battle(player, ws)
            return

        # Matched! Find which room this player was placed in
        room = None
        for r in self.rooms.values():
            if player.player_id in [
                p.player_id
                for p in r.players.values()
                if p.ws is player.ws
            ]:
                room = r
                break

        if room is None:
            await ws.close()
            return

        await self._run_player(room, player)

    async def _start_ai_battle(self, player: Player, ws: WebSocket):
        room_id = uuid.uuid4().hex[:8]
        room = Room(room_id=room_id)
        self.rooms[room_id] = room

        player.player_id = 1
        room.players[1] = player

        # Create AI player placeholder (character generated in background)
        ai_player = Player(ws=None, player_id=2)
        ai_player.ready = True
        room.players[2] = ai_player

        # Send to camera immediately
        await ws.send_json({
            "type": "joined",
            "player_id": 1,
            "players_in_room": 2,
        })
        await ws.send_json({
            "type": "both_joined",
            "player_id": 1,
        })

        # Generate AI character in background while player takes photo
        async def gen_ai_character():
            try:
                ai_char = await gemini_service.generate_random_character()
            except Exception:
                ai_char = {
                    "name": "謎の挑戦者",
                    "hp": 120, "attack": 60, "defense": 50, "speed": 55,
                    "special_move": "ミステリアスブロー",
                    "attribute": "打撃", "power": 50,
                    "description": "正体不明の挑戦者。油断はできない。",
                }
            try:
                ai_image = await gemini_service.generate_character_image(ai_char)
                ai_char["image"] = ai_image
            except Exception:
                pass
            ai_player.character = ai_char

        asyncio.create_task(gen_ai_character())

        await self._run_player(room, player)

    async def _run_player(self, room: Room, player: Player):
        try:
            while True:
                data = await player.ws.receive_text()
                msg = json.loads(data)
                await self.handle_message(room, player, msg)
        except Exception:
            pass
        finally:
            await self._handle_disconnect(room, player)

    async def handle_message(self, room: Room, player: Player, msg: dict):
        msg_type = msg.get("type")

        if msg_type == "image_submit":
            await self._handle_image_submit(room, player, msg)
        elif msg_type == "ready":
            await self._handle_ready(room, player)

    async def _handle_image_submit(self, room: Room, player: Player, msg: dict):
        player.image_data = msg.get("image_data", "")

        try:
            # Step 1: Analyze the object in the image (attribute & power)
            object_info = await gemini_service.analyze_object(player.image_data)

            # Step 2: Generate character using object analysis as context
            character = await gemini_service.analyze_image(
                player.image_data, object_info=object_info
            )

            # Ensure attribute and power are included in the character
            if "attribute" not in character:
                character["attribute"] = object_info.get("attribute", "打撃")
            if "power" not in character:
                character["power"] = object_info.get("power", 50)

            # Generate character image in parallel-ish: send stats first, then image
            await player.ws.send_json({
                "type": "character_generated",
                "character": character,
            })

            # Generate character illustration
            try:
                image_data_url = await gemini_service.generate_character_image(character)
                character["image"] = image_data_url
                await player.ws.send_json({
                    "type": "character_image",
                    "image": image_data_url,
                })
            except Exception:
                # Image generation is optional; continue without it
                pass

            player.character = character

            # Notify opponent that this player's character is ready
            for p in room.players.values():
                if p.player_id != player.player_id and p.ws is not None:
                    try:
                        await p.ws.send_json({"type": "opponent_character_ready"})
                    except Exception:
                        pass

        except Exception as e:
            await player.ws.send_json({
                "type": "error",
                "message": f"キャラクター生成に失敗しました: {str(e)}",
            })

    async def _handle_ready(self, room: Room, player: Player):
        player.ready = True

        # Notify opponent
        for p in room.players.values():
            if p.player_id != player.player_id and p.ws is not None:
                try:
                    await p.ws.send_json({"type": "opponent_ready"})
                except Exception:
                    pass

        # Check if both players are ready
        all_ready = (
            len(room.players) == 2
            and all(p.ready and p.character for p in room.players.values())
        )
        if all_ready:
            await self._start_battle(room)

    async def _start_battle(self, room: Room):
        room.state = "playing"

        players = list(room.players.values())
        p1, p2 = players[0], players[1]

        for p in room.players.values():
            if p.ws is not None:
                try:
                    await p.ws.send_json({
                        "type": "battle_start",
                        "player1": {"player_id": p1.player_id, "character": p1.character},
                        "player2": {"player_id": p2.player_id, "character": p2.character},
                    })
                except Exception:
                    pass

        try:
            result, _ = await asyncio.gather(
                gemini_service.resolve_battle(p1.character, p2.character),
                asyncio.sleep(5),
            )

            winner_player_id = p1.player_id if result["winner"] == 1 else p2.player_id

            battle_result = {
                "type": "battle_result",
                "winner_player_id": winner_player_id,
                "reason": result.get("reason", ""),
                "player1": {
                    "player_id": p1.player_id,
                    "character": p1.character,
                },
                "player2": {
                    "player_id": p2.player_id,
                    "character": p2.character,
                },
            }

            room.state = "finished"
            for p in room.players.values():
                if p.ws is not None:
                    try:
                        await p.ws.send_json(battle_result)
                    except Exception:
                        pass

        except Exception as e:
            for p in room.players.values():
                if p.ws is not None:
                    try:
                        await p.ws.send_json({
                            "type": "error",
                            "message": f"バトル処理に失敗しました: {str(e)}",
                        })
                    except Exception:
                        pass

    async def _handle_disconnect(self, room: Room, player: Player):
        room.players.pop(player.player_id, None)

        # Notify remaining player
        for p in room.players.values():
            if p.ws is not None:
                try:
                    await p.ws.send_json({"type": "opponent_disconnected"})
                except Exception:
                    pass

        # Clean up empty rooms
        if not room.players:
            self.rooms.pop(room.room_id, None)
