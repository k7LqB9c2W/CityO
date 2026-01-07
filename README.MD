Goal of this is to create a city builder game but with super simplified agents, but nice graphics. To allow 
creation of cities on a map the size 4300 square miles / 11137 square kilometers.

Working title: MegaCity Painter (Cities: Skylines vibe, minimal simulation)

One-line pitch:
Build a visually stunning, Cities: Skylines-style metropolis across an ultra-large region (about 4,300 square miles) using roads that generate zoning, with massive scale and smooth performance, but without heavy citizen/vehicle agent simulation.

Core fantasy

I’m painting and shaping a giant realistic city. It looks alive and dense from far away, but the engine stays fast because we fake complexity instead of simulating every person.

What the player does (core loop)

Lay roads (primary tool).

Roads generate a zoning grid and zoning eligibility.

Player paints zones (Residential, Commercial, Industrial, Mixed, Office).

Buildings spawn and upgrade based on zone type + land value + services access.

Player adds services and infrastructure (power, water, sewage, schools, fire, police, transit as “coverage,” not agents).

Metrics respond (population, jobs, revenue, traffic heatmaps, land value, happiness).

Expand outward, densify cores, create districts, hit visual and metric milestones.

Big constraints (non-negotiables)

Map scale: Region-scale map around 4,300 sq mi. Use streaming and chunking.

Performance first: Support huge cities by using instancing, LODs, impostors, and chunk-based updates.

Minimal simulation: No per-citizen pathfinding. No thousands of individual cars making decisions.

“Alive” illusion: Life is shown via lightweight systems: traffic flows as heatmaps, procedural crowds as decals/particles, day-night, lighting, ambient animations.

What “great graphics” means (make it concrete)

Modern lighting (PBR materials, shadows, GI if feasible, day/night cycle).

LOD pipeline: far buildings become simplified meshes or impostors.

Streets feel detailed: decals, lane markings, props, trees, streetlights, curb cuts.

Weather and atmosphere (fog/haze for distance, optional rain).

District identity through color grading, props, and building sets.

Simulation model (lightweight, not agents)


Traffic: Flow-based, not agent-based:

Maintain road segment capacities and compute congestion via aggregate volume.

Display results as heatmaps, travel-time overlays, and “bottleneck alerts.”

Services: Coverage fields (radius, capacity) that feed land value/happiness.

Buildings: Each building is a state machine:

states: empty lot → construction → occupied →


Key systems (MVP order)

Terrain + chunked world + camera + UI basics

Road tool + intersections + zoning grid generation from roads

Zoning paint tool + building spawning (simple procedural)

LOD/instancing + chunk streaming

Demand + budget loop


Traffic flow heatmap (aggregate)


Non-goals (explicitly tell the AI what not to build)

No Services coverage + land value heatmaps

No full citizen life simulation.

No individual car pathfinding at city scale.

No complex interior building simulation.

No “every object is physics” sandbox.

No multiplayer for now.

Tech direction assumptions (so the AI stays consistent)

Use chunked data model for the region.

Deterministic seeds for procedural buildings/props.

Rendering: GPU instancing everywhere possible.

Data: keep city state lightweight (tile/lot-based + building instances).

Deliverable expectation for the AI

When you code, always prioritize:

performance and chunking,

clean modular systems,

simple models that scale,

tools that feel good (roads + zoning + bulldoze + upgrade),

visuals via LODs and instancing.