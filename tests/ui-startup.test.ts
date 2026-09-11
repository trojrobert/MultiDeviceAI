import assert from "node:assert/strict";
import { test } from "node:test";
import { RoomController } from "../src/runtime/room.ts";

test("RoomController construction does not emit before the UI is mounted", () => {
  let emissions = 0;
  const room = new RoomController({
    onChange: () => {
      emissions++;
    },
  });

  assert.equal(emissions, 0);
  assert.equal(room.snapshot.role, undefined);
  assert.equal(room.snapshot.status, "Create a room or join one from a shared link.");
});
