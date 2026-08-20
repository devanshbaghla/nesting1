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
