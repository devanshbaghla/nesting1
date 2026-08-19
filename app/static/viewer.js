/* Interactive isometric preview of the uploaded part, with its noise picked out.
 *
 * The part and each noise fragment arrive as separate triangle soups so they
 * can be coloured, counted and hidden as groups. Normals are computed here
 * rather than sent, which halves the payload.
 *
 * three.js is vendored under /static/vendor so this works with no network.
 */
import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';

const PART_COLOUR  = 0x6ea8fe;   // calm blue: the thing you are keeping
const NOISE_COLOUR = 0xff4d4f;   // red: the thing the rule wants to delete

export class PartViewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({canvas, antialias: true});
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x11151c);

    this.camera = new THREE.PerspectiveCamera(35, 1, 0.1, 1e6);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;

    // three-point-ish lighting so a grey part still reads as a solid
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x202832, 1.5));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(1, 1.4, 1);
    this.scene.add(key);
    const rim = new THREE.DirectionalLight(0x88aaff, 0.5);
    rim.position.set(-1, -0.6, -0.8);
    this.scene.add(rim);

    this.root = new THREE.Group();
    this.scene.add(this.root);
    this.partMesh = null;
    this.noiseMeshes = [];
    this.markers = new THREE.Group();
    this.root.add(this.markers);

    this._resize = this._resize.bind(this);
    window.addEventListener('resize', this._resize);
    this._tick = this._tick.bind(this);
    requestAnimationFrame(this._tick);
  }

  _resize() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  _tick() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    requestAnimationFrame(this._tick);
  }

  static _geometry(positions) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position',
      new THREE.BufferAttribute(new Float32Array(positions), 3));
    g.computeVertexNormals();
    return g;
  }

  clear() {
    for (const m of [this.partMesh, ...this.noiseMeshes]) {
      if (!m) continue;
      this.root.remove(m);
      m.geometry.dispose();
      m.material.dispose();
    }
    this.partMesh = null;
    this.noiseMeshes = [];
    this.markers.clear();
  }

  /** Draw a payload from /api/preview/{id}/geometry. */
  load(payload) {
    this.clear();
    this._bounds = payload.bounds;

    if (payload.part && payload.part.length) {
      this.partMesh = new THREE.Mesh(
        PartViewer._geometry(payload.part),
        new THREE.MeshStandardMaterial({
          color: PART_COLOUR, roughness: 0.55, metalness: 0.1,
          flatShading: false, side: THREE.DoubleSide,
        }));
      this.root.add(this.partMesh);
    }

    for (const frag of (payload.noise || [])) {
      if (!frag.positions.length) continue;
      const mesh = new THREE.Mesh(
        PartViewer._geometry(frag.positions),
        new THREE.MeshStandardMaterial({
          color: NOISE_COLOUR, roughness: 0.4, metalness: 0.0,
          emissive: NOISE_COLOUR, emissiveIntensity: 0.35,
          side: THREE.DoubleSide,
        }));
      mesh.userData.fragment = frag.index;
      this.noiseMeshes.push(mesh);
      this.root.add(mesh);
      this._ring(mesh);
    }

    this._frame(payload.bounds);
  }

  /** A wire box around a noise fragment — a few red triangles are easy to miss. */
  _ring(mesh) {
    mesh.geometry.computeBoundingBox();
    const box = mesh.geometry.boundingBox;
    const size = new THREE.Vector3(); box.getSize(size);
    const centre = new THREE.Vector3(); box.getCenter(centre);
    const pad = Math.max(size.x, size.y, size.z) * 0.6 + 1e-6;
    const helper = new THREE.Box3Helper(
      new THREE.Box3(
        centre.clone().sub(size.clone().multiplyScalar(0.5)).subScalar(pad),
        centre.clone().add(size.clone().multiplyScalar(0.5)).addScalar(pad)),
      new THREE.Color(NOISE_COLOUR));
    this.markers.add(helper);
  }

  /** Isometric framing: the standard 3/4 view, sized to whatever was loaded. */
  _frame(bounds) {
    const [lo, hi] = bounds;
    const centre = new THREE.Vector3(
      (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2);
    const span = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) || 1;

    // STL is Z-up, three.js is Y-up; rotating the group beats rotating the
    // camera because the orbit target then stays where the user expects
    this.root.rotation.set(-Math.PI / 2, 0, 0);
    this.root.position.set(0, 0, 0);
    for (const m of [this.partMesh, ...this.noiseMeshes]) {
      if (m) m.position.set(-centre.x, -centre.y, -centre.z);
    }
    this.markers.position.set(-centre.x, -centre.y, -centre.z);

    const d = span * 1.9;
    this.camera.position.set(d, d * 0.82, d);
    this.camera.near = span / 500;
    this.camera.far = span * 100;
    this.controls.target.set(0, 0, 0);
    this.controls.update();
    this._resize();
  }

  setNoiseVisible(on) {
    for (const m of this.noiseMeshes) m.visible = on;
    this.markers.visible = on;
  }

  setPartOpacity(alpha) {
    if (!this.partMesh) return;
    this.partMesh.material.transparent = alpha < 1;
    this.partMesh.material.opacity = alpha;
    this.partMesh.material.needsUpdate = true;
  }

  resetView() { this._frame(this._bounds || [[-1, -1, -1], [1, 1, 1]]); }
}
