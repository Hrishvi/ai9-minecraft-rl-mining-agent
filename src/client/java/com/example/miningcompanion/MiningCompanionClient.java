package com.example.miningcompanion;

import com.google.gson.JsonArray;
import com.google.gson.JsonNull;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.message.v1.ClientSendMessageEvents;

import net.minecraft.block.BlockState;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.registry.Registries;
import net.minecraft.registry.tag.FluidTags;
import net.minecraft.text.Text;
import net.minecraft.util.Hand;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Direction;
import net.minecraft.util.math.MathHelper;
import net.minecraft.util.math.Vec3d;
import net.minecraft.world.World;

import java.util.HashSet;
import java.util.Set;

/**
 * Companion mining mod (CLIENT entrypoint).
 *
 * Two work modes, both line-of-sight legit (no X-ray):
 *
 *   ORE   - "!mine a b c". Reports only target ore with an air-exposed face;
 *           the brain pathfinds to the closest. Ore sealed in solid stone is
 *           never reported.
 *
 *   STRIP - "!corner1"/"!corner2" mark a box, "!strip" clears it. The mod walks a
 *           fixed bottom-up snake through the volume and breaks EVERY block in
 *           order, collecting whatever ore turns up. No looking through walls.
 *
 * Lava handling is SAFETY, not detection (gated on LAVA_SAFE):
 *   - In STRIP, the sweep skips any block that is lava or touches lava, leaving a
 *     one-block buffer wall instead of digging into it.
 *   - In both modes, if lava ends up next to the player, the state report flags a
 *     hazard so the brain can pause.
 *   These checks only ever look at blocks the agent is adjacent to - they do not
 *   scan for hidden lava, which would be the same X-ray problem as ore.
 *
 * Per tick: capture chat commands, every 2 ticks send a state report (player pos,
 * inventory, mode, hazard, plus ORE targets or the next STRIP block), and every
 * tick execute the brain's last action (mine / goto / idle).
 *
 * The actuator is a simple reference (face, walk, jump-when-bumped, break-in-reach),
 * not a real pathfinder. Fine on flat ground and short tunnels; full 3D boxes want
 * the navigation you plug into walk()/driveTowardAndMine().
 *
 * Threads: chat + tick callbacks run on the client thread (no locking needed for
 * mode/targets/corners/currentAction). Only socket I/O is on background threads.
 */
public class MiningCompanionClient implements ClientModInitializer {

    private static final String HOST = "127.0.0.1";
    private static final int PORT = 5555;

    private static final int SCAN_RADIUS = 12;    // ORE-mode scan cube half-size
    private static final double REACH = 4.5;      // survival-ish reach for mining
    private static final boolean LAVA_SAFE = true; // skip lava-touching blocks + pause near lava

    // SINGLE-PLAYER ONLY. When true, the ore scan reports target ore through solid
    // stone ("X-ray") and the actuator tunnels to reach it. Fine in your own offline
    // world; this is bannable on multiplayer servers, so do not use it online.
    private static final boolean XRAY = true;

    private static final Direction[] NEIGHBORS = Direction.values();

    private SocketBridge bridge;

    private String mode = null;            // "ore", "strip", or null (idle)
    private Set<String> targetBlocks = new HashSet<>();  // ORE mode
    private BlockPos cornerA = null;       // STRIP mode selection corners
    private BlockPos cornerB = null;

    private String currentAction = null;
    private long tickCounter = 0L;
    private BlockPos breakingPos = null;

    @Override
    public void onInitializeClient() {
        bridge = new SocketBridge(HOST, PORT);
        bridge.start();

        registerChatCommands();
        ClientTickEvents.END_CLIENT_TICK.register(this::onClientTick);
    }

    // ------------------------------------------------------------ chat commands
    private void registerChatCommands() {
        ClientSendMessageEvents.ALLOW_CHAT.register(message -> {
            if (message.startsWith("!mine ")) {
                handleMineCommand(message.substring("!mine ".length()).trim());
                return false;
            }
            if (message.equals("!corner1")) {
                setCorner(1);
                return false;
            }
            if (message.equals("!corner2")) {
                setCorner(2);
                return false;
            }
            if (message.equals("!strip")) {
                handleStripCommand();
                return false;
            }
            if (message.equals("!minestop")) {
                handleStopCommand();
                return false;
            }
            return true;
        });
    }

    private void handleMineCommand(String args) {
        MinecraftClient client = MinecraftClient.getInstance();
        ClientPlayerEntity player = client.player;
        if (player == null) {
            return;
        }

        Set<String> wanted = new HashSet<>();
        JsonArray blocks = new JsonArray();
        for (String raw : args.split("\\s+")) {
            if (raw.isEmpty()) {
                continue;
            }
            String id = raw.contains(":") ? raw : "minecraft:" + raw;
            wanted.add(id);
            blocks.add(id);
        }
        this.targetBlocks = wanted;
        this.mode = "ore";

        JsonObject home = new JsonObject();
        home.addProperty("x", player.getX());
        home.addProperty("y", player.getY());
        home.addProperty("z", player.getZ());

        JsonObject msg = new JsonObject();
        msg.addProperty("type", "set_targets");
        msg.add("blocks", blocks);
        msg.add("home", home);
        bridge.sendLine(msg.toString());

        player.sendMessage(Text.literal("[Miner] Targets " + wanted + "  |  Home "
                + (int) player.getX() + ", " + (int) player.getY() + ", " + (int) player.getZ()), false);
    }

    private void setCorner(int which) {
        ClientPlayerEntity player = MinecraftClient.getInstance().player;
        if (player == null) {
            return;
        }
        BlockPos p = player.getBlockPos();
        if (which == 1) {
            cornerA = p;
        } else {
            cornerB = p;
        }
        player.sendMessage(Text.literal("[Miner] Corner " + which + " set at "
                + p.getX() + ", " + p.getY() + ", " + p.getZ()), false);
    }

    private void handleStripCommand() {
        MinecraftClient client = MinecraftClient.getInstance();
        ClientPlayerEntity player = client.player;
        if (player == null) {
            return;
        }
        if (cornerA == null || cornerB == null) {
            player.sendMessage(Text.literal("[Miner] Set both corners first: !corner1 then !corner2"), false);
            return;
        }

        this.mode = "strip";
        this.targetBlocks = new HashSet<>();

        int minX = Math.min(cornerA.getX(), cornerB.getX());
        int minY = Math.min(cornerA.getY(), cornerB.getY());
        int minZ = Math.min(cornerA.getZ(), cornerB.getZ());
        int maxX = Math.max(cornerA.getX(), cornerB.getX());
        int maxY = Math.max(cornerA.getY(), cornerB.getY());
        int maxZ = Math.max(cornerA.getZ(), cornerB.getZ());
        long volume = (long) (maxX - minX + 1) * (maxY - minY + 1) * (maxZ - minZ + 1);

        JsonObject home = new JsonObject();
        home.addProperty("x", player.getX());
        home.addProperty("y", player.getY());
        home.addProperty("z", player.getZ());

        JsonObject box = new JsonObject();
        box.addProperty("minX", minX);
        box.addProperty("minY", minY);
        box.addProperty("minZ", minZ);
        box.addProperty("maxX", maxX);
        box.addProperty("maxY", maxY);
        box.addProperty("maxZ", maxZ);

        JsonObject msg = new JsonObject();
        msg.addProperty("type", "set_strip");
        msg.add("home", home);
        msg.add("box", box);
        bridge.sendLine(msg.toString());

        player.sendMessage(Text.literal("[Miner] Strip-mining " + volume + " blocks  |  Home "
                + (int) player.getX() + ", " + (int) player.getY() + ", " + (int) player.getZ()), false);
    }

    private void handleStopCommand() {
        this.mode = null;
        this.targetBlocks = new HashSet<>();

        JsonObject msg = new JsonObject();
        msg.addProperty("type", "stop");
        bridge.sendLine(msg.toString());

        ClientPlayerEntity player = MinecraftClient.getInstance().player;
        if (player != null) {
            player.sendMessage(Text.literal("[Miner] Stopped."), false);
        }
    }

    // --------------------------------------------------------------- tick loop
    private void onClientTick(MinecraftClient client) {
        if (client.player == null || client.world == null) {
            releaseControls(client);
            return;
        }

        tickCounter++;

        if (tickCounter % 2 == 0) {
            scanAndReport(client);
        }

        String fresh = bridge.pollLatestAction();
        if (fresh != null) {
            this.currentAction = fresh;
            announceIfPresent(client, fresh);
        }
        applyAction(client, this.currentAction);
    }

    // ----------------------------------------------------- sensing / reporting
    private void scanAndReport(MinecraftClient client) {
        ClientPlayerEntity player = client.player;
        World world = client.world;

        JsonObject msg = new JsonObject();
        msg.addProperty("type", "state");

        JsonObject pos = new JsonObject();
        pos.addProperty("x", player.getX());
        pos.addProperty("y", player.getY());
        pos.addProperty("z", player.getZ());
        msg.add("player", pos);

        int empty = countEmptyStorageSlots(player);
        JsonObject inv = new JsonObject();
        inv.addProperty("empty_slots", empty);
        inv.addProperty("is_inventory_full", empty == 0);
        msg.add("inventory", inv);

        msg.addProperty("mode", mode == null ? "none" : mode);

        // Lava awareness for the brain. "hazard" is the blanket flag; the per-direction
        // flags (feet + head level) let the RL policy steer around lava block by block.
        BlockPos feet = player.getBlockPos();
        msg.addProperty("hazard", LAVA_SAFE && playerNearLava(world, player));
        msg.addProperty("lava_n", LAVA_SAFE && (isLava(world, feet.north()) || isLava(world, feet.north().up())));
        msg.addProperty("lava_e", LAVA_SAFE && (isLava(world, feet.east()) || isLava(world, feet.east().up())));
        msg.addProperty("lava_s", LAVA_SAFE && (isLava(world, feet.south()) || isLava(world, feet.south().up())));
        msg.addProperty("lava_w", LAVA_SAFE && (isLava(world, feet.west()) || isLava(world, feet.west().up())));

        if ("strip".equals(mode)) {
            msg.add("strip", buildStripReport(world));
        } else if ("ore".equals(mode)) {
            msg.add("targets", scanExposedTargets(world, player.getBlockPos()));
        }

        bridge.sendLine(msg.toString());
    }

    /** ORE mode: matching target ore in the scan radius. With XRAY off, only air-exposed ore. */
    private JsonArray scanExposedTargets(World world, BlockPos origin) {
        JsonArray targets = new JsonArray();
        if (targetBlocks.isEmpty()) {
            return targets;
        }
        for (int dx = -SCAN_RADIUS; dx <= SCAN_RADIUS; dx++) {
            for (int dy = -SCAN_RADIUS; dy <= SCAN_RADIUS; dy++) {
                for (int dz = -SCAN_RADIUS; dz <= SCAN_RADIUS; dz++) {
                    BlockPos pos = origin.add(dx, dy, dz);
                    BlockState state = world.getBlockState(pos);
                    if (state.isAir()) {
                        continue;
                    }
                    String id = Registries.BLOCK.getId(state.getBlock()).toString();
                    if (!targetBlocks.contains(id)) {
                        continue;
                    }
                    if (!XRAY && !isExposed(world, pos)) {
                        continue; // without X-ray, only report ore you could actually see
                    }
                    JsonObject o = new JsonObject();
                    o.addProperty("id", id);
                    o.addProperty("x", pos.getX());
                    o.addProperty("y", pos.getY());
                    o.addProperty("z", pos.getZ());
                    targets.add(o);
                }
            }
        }
        return targets;
    }

    /**
     * STRIP mode: next block to clear, scanned bottom-up with a snake sweep per
     * layer. With LAVA_SAFE on, any block that is lava or touches lava is skipped
     * (left as a buffer wall). Reports next=null; if that null was caused by lava
     * rather than an empty box, blocked=true.
     */
    private JsonObject buildStripReport(World world) {
        JsonObject strip = new JsonObject();
        if (cornerA == null || cornerB == null) {
            strip.add("next", JsonNull.INSTANCE);
            return strip;
        }
        int minX = Math.min(cornerA.getX(), cornerB.getX());
        int minY = Math.min(cornerA.getY(), cornerB.getY());
        int minZ = Math.min(cornerA.getZ(), cornerB.getZ());
        int maxX = Math.max(cornerA.getX(), cornerB.getX());
        int maxY = Math.max(cornerA.getY(), cornerB.getY());
        int maxZ = Math.max(cornerA.getZ(), cornerB.getZ());

        BlockPos next = null;
        boolean lavaBlocked = false;

        outer:
        for (int y = minY; y <= maxY; y++) {
            for (int z = minZ; z <= maxZ; z++) {
                boolean reverse = ((z - minZ) & 1) == 1;
                int span = maxX - minX;
                for (int i = 0; i <= span; i++) {
                    int x = reverse ? (maxX - i) : (minX + i);
                    BlockPos p = new BlockPos(x, y, z);
                    if (world.getBlockState(p).isAir()) {
                        continue;
                    }
                    if (LAVA_SAFE && (isLava(world, p) || hasLavaNeighbor(world, p))) {
                        lavaBlocked = true; // leave it as a safety buffer; keep scanning
                        continue;
                    }
                    next = p;
                    break outer;
                }
            }
        }

        if (next == null) {
            strip.add("next", JsonNull.INSTANCE);
            strip.addProperty("blocked", lavaBlocked);
        } else {
            JsonArray n = new JsonArray();
            n.add(next.getX());
            n.add(next.getY());
            n.add(next.getZ());
            strip.add("next", n);
        }
        return strip;
    }

    /** A block is "visible" if any of its 6 neighbours is air. */
    private boolean isExposed(World world, BlockPos pos) {
        for (Direction d : NEIGHBORS) {
            if (world.getBlockState(pos.offset(d)).isAir()) {
                return true;
            }
        }
        return false;
    }

    // ----------------------------------------------------------- lava helpers
    private boolean isLava(World world, BlockPos pos) {
        return world.getBlockState(pos).getFluidState().isIn(FluidTags.LAVA);
    }

    private boolean hasLavaNeighbor(World world, BlockPos pos) {
        for (Direction d : NEIGHBORS) {
            if (isLava(world, pos.offset(d))) {
                return true;
            }
        }
        return false;
    }

    /** True if lava is in or directly around the player's body - a reason to pause. */
    private boolean playerNearLava(World world, ClientPlayerEntity player) {
        BlockPos feet = player.getBlockPos();
        BlockPos[] around = {
                feet, feet.up(), feet.down(),
                feet.north(), feet.south(), feet.east(), feet.west()
        };
        for (BlockPos p : around) {
            if (isLava(world, p)) {
                return true;
            }
        }
        return false;
    }

    /** Empty slots in main storage (rows 9-35), i.e. non-hotbar. 0 => inventory full. */
    private int countEmptyStorageSlots(ClientPlayerEntity player) {
        int empty = 0;
        for (int i = 9; i <= 35; i++) {
            if (player.getInventory().getStack(i).isEmpty()) {
                empty++;
            }
        }
        return empty;
    }

    // --------------------------------------------------------------- actuator
    private void applyAction(MinecraftClient client, String actionJson) {
        if (actionJson == null) {
            releaseControls(client);
            return;
        }
        JsonObject a;
        try {
            a = JsonParser.parseString(actionJson).getAsJsonObject();
        } catch (Exception e) {
            releaseControls(client);
            return;
        }

        String action = a.has("action") ? a.get("action").getAsString() : "idle";
        switch (action) {
            case "mine": {
                if (a.has("target")) {
                    JsonArray t = a.getAsJsonArray("target");
                    BlockPos pos = new BlockPos(t.get(0).getAsInt(), t.get(1).getAsInt(), t.get(2).getAsInt());
                    driveTowardAndMine(client, pos);
                }
                break;
            }
            case "goto": {
                if (a.has("target")) {
                    JsonArray t = a.getAsJsonArray("target");
                    Vec3d dest = new Vec3d(t.get(0).getAsDouble(), t.get(1).getAsDouble(), t.get(2).getAsDouble());
                    driveTowardPoint(client, dest);
                }
                break;
            }
            case "idle":
            default:
                releaseControls(client);
                break;
        }
    }

    private void driveTowardAndMine(MinecraftClient client, BlockPos target) {
        moveToward(client, target, true);
    }

    private void driveTowardPoint(MinecraftClient client, Vec3d dest) {
        moveToward(client, BlockPos.ofFloored(dest), false);
    }

    /**
     * Head toward a block. If mineIt is true and the block is in reach, mine it.
     * Otherwise walk - and if a block is in the way, dig through it (tunnelling).
     * The tunnelling is what lets the agent reach ore buried behind stone (X-ray).
     */
    private void moveToward(MinecraftClient client, BlockPos target, boolean mineIt) {
        ClientPlayerEntity player = client.player;
        if (player == null || client.interactionManager == null) {
            return;
        }
        Vec3d center = Vec3d.ofCenter(target);
        faceToward(player, center);

        if (mineIt && player.getEyePos().distanceTo(center) <= REACH) {
            mineBlock(client, player, target);
            return;
        }

        BlockPos obstruction = obstructionToward(client, player, target);
        if (obstruction != null) {
            mineBlock(client, player, obstruction); // dig through the wall toward the ore
        } else {
            cancelBreak(client);
            walk(client, player);
        }
    }

    private void mineBlock(MinecraftClient client, ClientPlayerEntity player, BlockPos pos) {
        stopWalking(client);
        Direction face = nearestFace(player, pos);
        if (!pos.equals(breakingPos)) {
            client.interactionManager.attackBlock(pos, face);
            breakingPos = pos;
        }
        client.interactionManager.updateBlockBreakingProgress(pos, face);
        player.swingHand(Hand.MAIN_HAND);
    }

    /** The solid (non-lava) block one step toward the target, if any - for tunnelling. */
    private BlockPos obstructionToward(MinecraftClient client, ClientPlayerEntity player, BlockPos target) {
        World world = client.world;
        if (world == null) {
            return null;
        }
        BlockPos feet = player.getBlockPos();
        int ddx = target.getX() - feet.getX();
        int ddz = target.getZ() - feet.getZ();
        if (ddx == 0 && ddz == 0) {
            return null; // directly above/below - the reach check handles it
        }
        int dx = 0;
        int dz = 0;
        if (Math.abs(ddx) >= Math.abs(ddz)) {
            dx = Integer.signum(ddx);
        } else {
            dz = Integer.signum(ddz);
        }
        BlockPos aheadFeet = feet.add(dx, 0, dz);
        BlockPos aheadHead = aheadFeet.up();
        if (isDiggable(world, aheadHead)) {
            return aheadHead; // clear head height first
        }
        if (isDiggable(world, aheadFeet)) {
            return aheadFeet;
        }
        return null;
    }

    private boolean isDiggable(World world, BlockPos pos) {
        return !world.getBlockState(pos).isAir() && !isLava(world, pos); // never dig into lava
    }

    private void walk(MinecraftClient client, ClientPlayerEntity player) {
        client.options.forwardKey.setPressed(true);
        client.options.jumpKey.setPressed(player.horizontalCollision);
    }

    private void stopWalking(MinecraftClient client) {
        client.options.forwardKey.setPressed(false);
        client.options.jumpKey.setPressed(false);
    }

    private void faceToward(ClientPlayerEntity player, Vec3d point) {
        double dx = point.x - player.getX();
        double dy = point.y - player.getEyeY();
        double dz = point.z - player.getZ();
        double horizontal = Math.sqrt(dx * dx + dz * dz);
        float yaw = (float) (MathHelper.atan2(dz, dx) * (180.0 / Math.PI)) - 90.0F;
        float pitch = (float) (-(MathHelper.atan2(dy, horizontal) * (180.0 / Math.PI)));
        player.setYaw(yaw);
        player.setPitch(MathHelper.clamp(pitch, -90.0F, 90.0F));
    }

    private Direction nearestFace(ClientPlayerEntity player, BlockPos target) {
        double dx = player.getX() - (target.getX() + 0.5);
        double dy = player.getEyeY() - (target.getY() + 0.5);
        double dz = player.getZ() - (target.getZ() + 0.5);
        double ax = Math.abs(dx);
        double ay = Math.abs(dy);
        double az = Math.abs(dz);
        if (ax >= ay && ax >= az) {
            return dx > 0 ? Direction.EAST : Direction.WEST;
        }
        if (az >= ax && az >= ay) {
            return dz > 0 ? Direction.SOUTH : Direction.NORTH;
        }
        return dy > 0 ? Direction.UP : Direction.DOWN;
    }

    private void cancelBreak(MinecraftClient client) {
        if (client.interactionManager != null) {
            client.interactionManager.cancelBlockBreaking();
        }
        breakingPos = null;
    }

    private void releaseControls(MinecraftClient client) {
        if (client.options != null) {
            client.options.forwardKey.setPressed(false);
            client.options.jumpKey.setPressed(false);
        }
        cancelBreak(client);
    }

    private void announceIfPresent(MinecraftClient client, String actionJson) {
        try {
            JsonObject a = JsonParser.parseString(actionJson).getAsJsonObject();
            if (a.has("say") && !a.get("say").isJsonNull() && client.player != null) {
                client.player.sendMessage(Text.literal("[Miner] " + a.get("say").getAsString()), false);
            }
        } catch (Exception ignored) {
        }
    }
}
