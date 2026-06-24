// Virtual hand viewer: renders a procedural hand and eases its finger curl
// toward the latest `closure` streamed by the myo-rt loop over WebSocket.
import * as THREE from "./vendor/three.module.js";

const PORT = new URLSearchParams(location.search).get("port") || "8765";

// --- scene ---------------------------------------------------------------
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);

const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.1, 100);
camera.position.set(0, 2, 9);
camera.lookAt(0, 1.3, 0);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(devicePixelRatio);
document.body.appendChild(renderer.domElement);

scene.add(new THREE.HemisphereLight(0xffffff, 0x223344, 1.1));
const key = new THREE.DirectionalLight(0xffffff, 1.0);
key.position.set(3, 5, 4);
scene.add(key);

const skin = new THREE.MeshStandardMaterial({ color: 0x6fb3d2, roughness: 0.5, metalness: 0.1 });

// --- hand geometry -------------------------------------------------------
const hand = new THREE.Group();
scene.add(hand);

const palm = new THREE.Mesh(new THREE.BoxGeometry(2.2, 2.4, 0.6), skin);
hand.add(palm);

// A finger is a chain of joint pivots; curling rotates every pivot, so each
// knuckle bends and the segments compound into a natural curl.
function makeFinger(segLens, radius) {
  const root = new THREE.Group();
  const joints = [];
  let parent = root;
  for (const len of segLens) {
    const pivot = new THREE.Group();
    parent.add(pivot);
    const seg = new THREE.Mesh(new THREE.CapsuleGeometry(radius, len, 4, 8), skin);
    seg.position.y = len / 2 + radius;
    pivot.add(seg);
    const next = new THREE.Group();
    next.position.y = len + radius * 2;
    pivot.add(next);
    joints.push(pivot);
    parent = next;
  }
  return { root, joints };
}

const fingers = [];
const defs = [
  { x: -0.85, segs: [0.7, 0.55, 0.45] },
  { x: -0.3, segs: [0.85, 0.6, 0.5] },
  { x: 0.3, segs: [0.8, 0.6, 0.5] },
  { x: 0.85, segs: [0.65, 0.5, 0.4] },
];
for (const d of defs) {
  const f = makeFinger(d.segs, 0.16);
  f.root.position.set(d.x, 1.2, 0);
  hand.add(f.root);
  fingers.push(f);
}
// Thumb: splayed off the lower-left of the palm.
const thumb = makeFinger([0.7, 0.55], 0.2);
thumb.root.position.set(-1.15, -0.3, 0.1);
thumb.root.rotation.z = Math.PI * 0.38;
hand.add(thumb.root);
fingers.push(thumb);

// --- closure animation ---------------------------------------------------
const MAX_CURL = 1.6; // radians per joint at closure = 1
let target = 0.15;
let cur = 0.15;

function setClosure(c) {
  target = Math.min(1, Math.max(0, c));
}

function animate() {
  requestAnimationFrame(animate);
  cur += (target - cur) * 0.15; // ease toward target
  for (const f of fingers) {
    for (const j of f.joints) j.rotation.x = cur * MAX_CURL;
  }
  hand.rotation.y = Math.sin(performance.now() * 0.0003) * 0.3; // gentle idle sway
  renderer.render(scene, camera);
}
animate();

// --- websocket -----------------------------------------------------------
const statusEl = document.getElementById("status");
const poseEl = document.getElementById("pose");

function connect() {
  const ws = new WebSocket(`ws://127.0.0.1:${PORT}`);
  ws.onopen = () => (statusEl.textContent = `connected · ws://127.0.0.1:${PORT}`);
  ws.onclose = () => {
    statusEl.textContent = "disconnected — retrying…";
    setTimeout(connect, 1000);
  };
  ws.onmessage = (e) => {
    try {
      const m = JSON.parse(e.data);
      if (typeof m.closure === "number") setClosure(m.closure);
      if (m.pose) poseEl.textContent = m.pose;
    } catch {
      /* ignore malformed frames */
    }
  };
}
connect();

addEventListener("resize", () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
