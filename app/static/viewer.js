/* The two interactive views this app shows, both on vendored three.js.
 *
 *   PartViewer   the uploaded part BEFORE nesting, with the fragments the
 *                noise rule would drop picked out in red, so the call can be
 *                seen and overridden before anything is deleted.
 *   PairViewer   one nested pair AFTER nesting, loaded from a GLB.
 *
 * Two classes rather than one configurable viewer, because they answer
 * different questions and take different input: PartViewer is handed
 * pre-split triangle soups it must colour by group, PairViewer is handed a GLB
 * that already carries a node and a colour per copy.
 *
 * PairViewer replaced seven server-rendered PNG viewpoints. Rendering them was
 * the most expensive stage in the pipeline after refinement and none of it fed
 * the result: on electric_drill.stl the three preview images cost 22.6 s
 * against the 0.15 s the GLB writes take, and on a 300k-face part one shaded
 * Poly3DCollection alone cost 232.7 s.
 *
 * The vendored addons keep upstream's directory layout, because GLTFLoader
 * imports '../utils/BufferGeometryUtils.js' by relative path -- flatten them
 * into one folder and that import 404s at runtime, with the only symptom being
 * a module that silently fails to load.
 */
import * as THREE from 'three';
import {OrbitControls} from './vendor/controls/OrbitControls.js';
import {GLTFLoader} from './vendor/loaders/GLTFLoader.js';

const PART_COLOUR  = 0x6ea8fe;   // calm blue: the thing you are keeping
const NOISE_COLOUR = 0xff4d4f;   // red: the thing the rule wants to delete
const PICK_COLOUR  = 0xffc53d;   // amber: the one body you are inspecting
const GHOST_ALPHA  = 0.14;       // faint enough to read through, still there

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
    this.clearSelection();
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

    this._centreScene(payload.bounds);
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

  /** Put the part's centre at the origin. Once per load, never on framing.
   *
   * Kept separate from the camera move so that focusing on a body shifts the
   * view and not the geometry — recomputing the offset per frame would slide
   * the whole scene under the user every time they clicked a row.
   */
  _centreScene(bounds) {
    const [lo, hi] = bounds;
    this._offset = new THREE.Vector3(
      -(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -(lo[2] + hi[2]) / 2);
    // STL is Z-up, three.js is Y-up; rotating the group beats rotating the
    // camera because the orbit target then stays where the user expects
    this.root.rotation.set(-Math.PI / 2, 0, 0);
    this.root.position.set(0, 0, 0);
    for (const m of [this.partMesh, ...this.noiseMeshes]) {
      if (m) m.position.copy(this._offset);
    }
    this.markers.position.copy(this._offset);
  }

  /** Isometric framing: the standard 3/4 view, sized to whatever is passed. */
  _frame(bounds) {
    const [lo, hi] = bounds;
    const span = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) || 1;
    const target = new THREE.Vector3(
      (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2)
      .add(this._offset || new THREE.Vector3())
      .applyEuler(this.root.rotation);

    const d = span * 1.9;
    this.camera.position.set(target.x + d, target.y + d * 0.82, target.z + d);
    // near/far track the whole part, not the selection, or focusing on a 0.4 mm
    // fragment would clip everything around it out of the scene
    const whole = this._bounds || bounds;
    const wide = Math.max(whole[1][0] - whole[0][0], whole[1][1] - whole[0][1],
                          whole[1][2] - whole[0][2]) || span;
    this.camera.near = Math.max(span / 500, wide / 1e5);
    this.camera.far = wide * 100;
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(target);
    this.controls.update();
    this._resize();
  }

  setNoiseVisible(on) {
    this._noiseWanted = on;
    for (const m of this.noiseMeshes) m.visible = on;
    this.markers.visible = on && !this.pick;
  }

  setPartOpacity(alpha) {
    this._partAlpha = alpha;
    if (!this.pick) this._applyAlpha(alpha);
  }

  _applyAlpha(alpha) {
    for (const m of [this.partMesh, ...this.noiseMeshes]) {
      if (!m) continue;
      m.material.transparent = alpha < 1;
      m.material.opacity = alpha;
      m.material.depthWrite = alpha >= 1;
      m.material.needsUpdate = true;
    }
  }

  /** Draw one body in the pick colour, with everything else left faintly on.
   *
   * The rest of the part is dimmed rather than hidden — a 0.4 mm fragment in a
   * 400 mm housing means nothing without the housing around it to place it.
   * The wire box is what actually makes it findable at that size.
   */
  selectBody(payload) {
    this.clearSelection();
    if (!payload) return;

    const geom = PartViewer._geometry(payload.positions);
    this.pick = new THREE.Mesh(geom, new THREE.MeshStandardMaterial({
      color: PICK_COLOUR, roughness: 0.35, metalness: 0.0,
      emissive: PICK_COLOUR, emissiveIntensity: 0.45,
      side: THREE.DoubleSide,
      // the body is already in the part buffer underneath; nudge this copy
      // toward the camera so the two do not fight over the same depth
      polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2,
    }));
    this.pick.position.copy(this._offset);
    this.root.add(this.pick);

    this.pickBox = new THREE.Group();
    this.pickBox.position.copy(this._offset);
    const [lo, hi] = payload.bounds;
    const box = new THREE.Box3(new THREE.Vector3(...lo), new THREE.Vector3(...hi));
    const size = new THREE.Vector3(); box.getSize(size);
    const pad = Math.max(size.x, size.y, size.z) * 0.35 + 1e-6;
    box.expandByScalar(pad);
    this.pickBox.add(new THREE.Box3Helper(box, new THREE.Color(PICK_COLOUR)));
    this.root.add(this.pickBox);
    this._pickBounds = payload.bounds;

    this._applyAlpha(GHOST_ALPHA);
    this.markers.visible = false;
  }

  clearSelection() {
    for (const o of [this.pick, this.pickBox]) {
      if (!o) continue;
      this.root.remove(o);
      o.traverse?.(c => { c.geometry?.dispose?.(); c.material?.dispose?.(); });
    }
    this.pick = this.pickBox = null;
    this._pickBounds = null;
    this._applyAlpha(this._partAlpha ?? 1);
    this.markers.visible = this._noiseWanted !== false;
  }

  /** Frame the selected body, or the whole part when nothing is selected. */
  focusSelection() {
    this._frame(this._pickBounds || this._bounds ||
                [[-1, -1, -1], [1, 1, 1]]);
  }

  resetView() { this._frame(this._bounds || [[-1, -1, -1], [1, 1, 1]]); }
}


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
