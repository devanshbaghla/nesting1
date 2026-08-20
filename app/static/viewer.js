/* Interactive iso view of a nested pair, loaded from a GLB.
 *
 * This replaced seven server-rendered PNG viewpoints. Rendering them was the
 * most expensive stage in the pipeline after refinement and none of it fed the
 * result, so the geometry now travels to the browser once and the camera moves
 * here instead. The default pose is the isometric the PNG used, so the first
 * frame looks like what the old image showed — the difference is that you can
 * drag it.
 *
 * Two copies arrive as separately-named, separately-coloured nodes (copy_A,
 * copy_B), which is what lets the interlock be read: rotate until the mating
 * faces line up and the two colours tell you which body is which.
 */
// The vendored addons keep upstream's directory layout, because GLTFLoader
// imports '../utils/BufferGeometryUtils.js' by relative path — flatten them
// into one folder and that import 404s at runtime, with the only symptom being
// a module that silently fails to load.
import * as THREE from 'three';
import {OrbitControls} from './vendor/controls/OrbitControls.js';
import {GLTFLoader} from './vendor/loaders/GLTFLoader.js';

const ISO_AZIMUTH = 35 * Math.PI / 180;
const ISO_ELEVATION = 24 * Math.PI / 180;
const FIT = 1.45;                 // how much room to leave around the part

/** One canvas, one model. Call dispose() when the canvas goes away. */
export class PairViewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({
      canvas, antialias: true, alpha: true, powerPreference: 'high-performance',
    });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.1, 1e5);

    // Three lights, not one: a single headlight flattens a machined face into
    // a solid block and the whole point is to read the surface relief.
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x33383d, 0.85));
    const key = new THREE.DirectionalLight(0xffffff, 1.5);
    key.position.set(1, 0.7, 1.4);
    this.scene.add(key);
    const rim = new THREE.DirectionalLight(0xcfe3ff, 0.5);
    rim.position.set(-1.2, -0.4, -0.8);
    this.scene.add(rim);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.09;
    this.controls.rotateSpeed = 0.85;

    this.root = null;
    this.bodies = [];
    this._raf = null;
    this._dirty = true;
    this.controls.addEventListener('change', () => this._wake());

    // A reflow of the card grid changes the aspect ratio, and the framing is
    // solved against it — so re-solve, unless the user has posed the view, in
    // which case yanking it back to iso would be the wrong kind of helpful.
    this._onResize = () => {
      this._resize();
      if (this.root && !this._touched) this._frameIso();
      this._wake();
    };
    this._touched = false;
    this.controls.addEventListener('start', () => { this._touched = true; });
    // ResizeObserver rather than a window listener: these canvases live in a
    // responsive card grid and change size when the grid reflows, which no
    // window resize event reports.
    this._ro = new ResizeObserver(this._onResize);
    this._ro.observe(canvas.parentElement || canvas);

    this._loop = this._loop.bind(this);
    this._raf = requestAnimationFrame(this._loop);
  }

  async load(url) {
    const gltf = await new GLTFLoader().loadAsync(url);
    if (this.root) this._disposeRoot();
    this.root = gltf.scene;
    this.bodies = [];
    this.root.traverse((o) => {
      if (!o.isMesh) return;
      o.geometry.computeVertexNormals();
      // The GLB carries one flat baseColorFactor per copy. Keep that colour but
      // swap in a material that reads as metal under these lights.
      const src = o.material;
      o.material = new THREE.MeshStandardMaterial({
        color: src && src.color ? src.color.clone() : new THREE.Color(0x8899a6),
        metalness: 0.15, roughness: 0.55, flatShading: false,
      });
      if (src && src.dispose) src.dispose();
      this.bodies.push(o);
    });
    this.scene.add(this.root);
    // resize first: the framing solves against camera.aspect, which _resize
    // is what sets, so doing this the other way round frames to a 1:1 canvas
    this._resize();
    this._frameIso();
    this._wake();
    return this.bodies.length;
  }

  /** Point the camera down the isometric axis and pull back to fit. */
  _frameIso() {
    // GLB is Y-up, the parts are modelled Z-up; tilting the model rather than
    // the camera keeps OrbitControls' vertical limits behaving normally.
    this.root.rotation.x = -Math.PI / 2;
    this.root.updateMatrixWorld(true);

    const box = new THREE.Box3().setFromObject(this.root);
    const size = box.getSize(new THREE.Vector3());
    const mid = box.getCenter(new THREE.Vector3());
    const radius = Math.max(size.length() * 0.5, 1e-4);

    // Fit the box's on-screen extent, not its bounding sphere. A nested pair is
    // usually long and thin — the reference part is 28 x 35 x 248 mm — and
    // sphere-fitting such a box leaves it floating in a third of the frame.
    const halfV = Math.tan(this.camera.fov * Math.PI / 360);
    const aspect = Math.max(this.camera.aspect || 1, 1e-3);
    // width across the iso view direction, height straight up
    const across = Math.hypot(size.x, size.z) * 0.5;
    const up = size.y * 0.5;
    const d = FIT * Math.max(up / halfV, across / (halfV * aspect));

    this.camera.position.set(
      mid.x + d * Math.cos(ISO_ELEVATION) * Math.cos(ISO_AZIMUTH),
      mid.y + d * Math.sin(ISO_ELEVATION),
      mid.z + d * Math.cos(ISO_ELEVATION) * Math.sin(ISO_AZIMUTH),
    );
    this.camera.near = Math.max(d / 5000, 0.01);
    this.camera.far = d * 12 + radius * 4;
    this.controls.target.copy(mid);
    this.controls.minDistance = radius * 0.15;
    this.controls.maxDistance = d * 6;
    this.controls.update();
    this._home = {pos: this.camera.position.clone(), target: mid.clone()};
  }

  resetView() {
    if (!this._home) return;
    this.camera.position.copy(this._home.pos);
    this.controls.target.copy(this._home.target);
    this.controls.update();
    this._wake();
  }

  /** Show one copy solid and the other translucent, or both solid. */
  isolate(index) {
    this.bodies.forEach((m, i) => {
      const dim = index !== null && index !== i;
      const was = m.material.transparent;
      m.material.transparent = dim;
      m.material.opacity = dim ? 0.16 : 1.0;
      m.material.depthWrite = !dim;
      // Flipping `transparent` changes which shader program the material
      // compiles to, so without this the opacity is set and simply ignored —
      // the label changes and the model does not.
      if (was !== dim) m.material.needsUpdate = true;
    });
    this._wake();
  }

  setSpin(on) { this._spin = on; this._wake(); }

  _resize() {
    const host = this.canvas.parentElement || this.canvas;
    const w = Math.max(host.clientWidth, 1);
    const h = Math.max(host.clientHeight, 1);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  // Render on demand rather than every frame: a results page can hold ten of
  // these, and ten idle render loops is a hot laptop for no reason.
  _wake() { this._dirty = true; }

  _loop() {
    this._raf = requestAnimationFrame(this._loop);
    if (this._spin && this.root) { this.root.rotation.z += 0.004; this._dirty = true; }
    const moving = this.controls.update();
    if (!this._dirty && !moving) return;
    this._dirty = false;
    this.renderer.render(this.scene, this.camera);
  }

  _disposeRoot() {
    this.root.traverse((o) => {
      if (!o.isMesh) return;
      o.geometry.dispose();
      if (o.material.dispose) o.material.dispose();
    });
    this.scene.remove(this.root);
    this.root = null;
  }

  dispose() {
    if (this._raf) cancelAnimationFrame(this._raf);
    this._ro.disconnect();
    this.controls.dispose();
    if (this.root) this._disposeRoot();
    this.renderer.dispose();
  }
}
