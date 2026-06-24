//! Browser transport for the virtual hand.
//!
//! `PoseBroadcaster` runs a tiny WebSocket server on a background thread and
//! pushes pose messages to any connected browsers. `WebHand` is the `Effector`
//! that sends `{pose, closure}` on each pose change. Everything here is
//! best-effort and non-blocking: no client, a slow client, or a dropped
//! connection never stalls the real-time loop.

use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::json;
use tracing::info;
use tungstenite::{Message, WebSocket};

use crate::MyoError;
use crate::effector::{Effector, HandPose};

type Clients = Arc<Mutex<Vec<WebSocket<TcpStream>>>>;

/// A WebSocket server that broadcasts text messages to connected clients.
pub struct PoseBroadcaster {
    clients: Clients,
    port: u16,
}

impl PoseBroadcaster {
    /// Bind `addr` (use `127.0.0.1:0` to let the OS pick a port) and start
    /// accepting clients on a background thread.
    pub fn bind(addr: &str) -> Result<Self, MyoError> {
        let listener =
            TcpListener::bind(addr).map_err(|e| MyoError::Effector(format!("bind {addr}: {e}")))?;
        let port = listener
            .local_addr()
            .map_err(|e| MyoError::Effector(e.to_string()))?
            .port();
        let clients: Clients = Arc::new(Mutex::new(Vec::new()));

        let accept_clients = clients.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                // Bound writes so a wedged client can't hang a broadcast.
                let _ = stream.set_write_timeout(Some(Duration::from_millis(500)));
                if let Ok(ws) = tungstenite::accept(stream) {
                    accept_clients.lock().unwrap().push(ws);
                }
            }
        });

        Ok(PoseBroadcaster { clients, port })
    }

    /// The bound port.
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Send `msg` to every connected client, dropping any that error.
    pub fn broadcast(&self, msg: &str) {
        let mut clients = self.clients.lock().unwrap();
        clients.retain_mut(|ws| ws.send(Message::text(msg)).is_ok());
    }
}

/// Effector that streams the current pose + closure to browsers.
pub struct WebHand {
    broadcaster: PoseBroadcaster,
    current: Option<String>,
}

impl WebHand {
    /// Bind the broadcaster on `addr` (e.g. `127.0.0.1:8765`).
    pub fn bind(addr: &str) -> Result<Self, MyoError> {
        Ok(WebHand {
            broadcaster: PoseBroadcaster::bind(addr)?,
            current: None,
        })
    }

    /// The bound port (useful for logging the viewer URL).
    pub fn port(&self) -> u16 {
        self.broadcaster.port()
    }
}

impl Effector for WebHand {
    fn apply(&mut self, pose: &HandPose) {
        let msg = json!({ "pose": pose.name, "closure": pose.closure() }).to_string();
        self.broadcaster.broadcast(&msg);
        if self.current.as_deref() != Some(pose.name.as_str()) {
            info!(to = %pose.name, closure = pose.closure(), "web hand pose change");
            self.current = Some(pose.name.clone());
        }
    }

    fn current(&self) -> Option<&str> {
        self.current.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effector::{Effector, HandPose};
    use std::net::TcpStream;
    use std::time::{Duration, Instant};
    use tungstenite::Message;
    use tungstenite::stream::MaybeTlsStream;

    /// Connect a ws client to `port` and set a short read timeout so polling
    /// reads don't block forever.
    fn connect(port: u16) -> tungstenite::WebSocket<MaybeTlsStream<TcpStream>> {
        let (sock, _resp) =
            tungstenite::connect(format!("ws://127.0.0.1:{port}")).expect("client connect");
        if let MaybeTlsStream::Plain(s) = sock.get_ref() {
            s.set_read_timeout(Some(Duration::from_millis(100)))
                .unwrap();
        }
        sock
    }

    /// Poll for the next text message, retrying `kick` (e.g. a re-broadcast)
    /// until one arrives or we time out. Tolerates the accept-thread race.
    fn read_text(
        sock: &mut tungstenite::WebSocket<MaybeTlsStream<TcpStream>>,
        mut kick: impl FnMut(),
    ) -> String {
        let deadline = Instant::now() + Duration::from_secs(3);
        while Instant::now() < deadline {
            kick();
            match sock.read() {
                Ok(Message::Text(t)) => return t.to_string(),
                _ => std::thread::sleep(Duration::from_millis(20)),
            }
        }
        panic!("no message received before deadline");
    }

    #[test]
    fn broadcaster_delivers_to_connected_client() {
        let b = PoseBroadcaster::bind("127.0.0.1:0").unwrap();
        let mut client = connect(b.port());
        let got = read_text(&mut client, || b.broadcast("hello"));
        assert_eq!(got, "hello");
    }

    #[test]
    fn web_hand_broadcasts_pose_and_closure() {
        let mut hand = WebHand::bind("127.0.0.1:0").unwrap();
        let mut client = connect(hand.port());
        let json = read_text(&mut client, || {
            hand.apply(&HandPose::from_class("hand_close"))
        });
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["pose"], "close");
        assert_eq!(v["closure"], 1.0);
    }
}
