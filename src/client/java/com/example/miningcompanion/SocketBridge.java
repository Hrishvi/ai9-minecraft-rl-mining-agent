package com.example.miningcompanion;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * A tiny line-delimited JSON bridge between this mod (the TCP client) and the
 * Python brain (the TCP server at 127.0.0.1:PORT).
 *
 *   - One background thread keeps the connection alive and auto-reconnects.
 *   - Outbound "state" lines are queued by the client thread, flushed by a
 *     dedicated writer thread.
 *   - Inbound "action" lines are queued, then drained by the client thread
 *     once per tick.
 *
 * None of this ever blocks the Minecraft client thread.
 */
public class SocketBridge {

    private final String host;
    private final int port;

    private final AtomicBoolean running = new AtomicBoolean(true);
    private final LinkedBlockingQueue<String> outgoing = new LinkedBlockingQueue<>();
    private final ConcurrentLinkedQueue<String> incoming = new ConcurrentLinkedQueue<>();

    // Access is guarded by 'this' so the net thread and writer thread agree on state.
    private volatile BufferedWriter writer;

    public SocketBridge(String host, int port) {
        this.host = host;
        this.port = port;
    }

    public void start() {
        Thread net = new Thread(this::connectLoop, "Miner-Bridge-Net");
        net.setDaemon(true);
        net.start();

        Thread wr = new Thread(this::writeLoop, "Miner-Bridge-Write");
        wr.setDaemon(true);
        wr.start();
    }

    /** Queue a JSON line to send. Dropped silently if disconnected/backed up (state is ephemeral). */
    public void sendLine(String json) {
        if (outgoing.size() < 256) {
            outgoing.add(json);
        }
    }

    /**
     * Returns the most recent action line received since the previous call, or null.
     * Older queued lines are discarded - we only ever care about the latest decision.
     */
    public String pollLatestAction() {
        String last = null;
        String s;
        while ((s = incoming.poll()) != null) {
            last = s;
        }
        return last;
    }

    public void stop() {
        running.set(false);
    }

    private void connectLoop() {
        while (running.get()) {
            try (Socket socket = new Socket(host, port)) {
                socket.setTcpNoDelay(true);
                synchronized (this) {
                    this.writer = new BufferedWriter(
                            new OutputStreamWriter(socket.getOutputStream(), StandardCharsets.UTF_8));
                }
                BufferedReader reader = new BufferedReader(
                        new InputStreamReader(socket.getInputStream(), StandardCharsets.UTF_8));

                String line;
                while (running.get() && (line = reader.readLine()) != null) {
                    incoming.add(line);
                }
            } catch (IOException ignored) {
                // Brain not running yet, or the link dropped - fall through and retry.
            } finally {
                synchronized (this) {
                    this.writer = null;
                }
            }
            sleep(2000L); // back off before reconnecting
        }
    }

    private void writeLoop() {
        while (running.get()) {
            try {
                String msg = outgoing.poll(1L, TimeUnit.SECONDS);
                if (msg == null) {
                    continue;
                }
                synchronized (this) {
                    if (writer != null) {
                        writer.write(msg);
                        writer.write("\n");
                        writer.flush();
                    }
                }
            } catch (Exception ignored) {
                // Write failed; connectLoop will rebuild the connection.
            }
        }
    }

    private static void sleep(long ms) {
        try {
            Thread.sleep(ms);
        } catch (InterruptedException ignored) {
            Thread.currentThread().interrupt();
        }
    }
}
