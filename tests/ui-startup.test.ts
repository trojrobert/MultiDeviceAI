import assert from "node:assert/strict";
import { test } from "node:test";
import { ClusterController } from "../src/runtime/cluster.ts";

test("ClusterController construction does not emit before the UI is mounted", () => {
  let emissions = 0;
  const cluster = new ClusterController({
    onChange: () => {
      emissions++;
    },
  });

  assert.equal(emissions, 0);
  assert.equal(cluster.snapshot.role, undefined);
  assert.equal(cluster.snapshot.status, "Create a cluster or join one from a shared link.");
});
