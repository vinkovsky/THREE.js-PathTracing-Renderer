precision highp float;
precision highp int;
precision highp sampler2D;

uniform sampler2D tTriangleTexture;
uniform sampler2D tAABBTexture;
uniform sampler2D tHDRTexture;

#include <pathtracing_uniforms_and_defines>

uniform vec3 uMaterialColor;
uniform vec3 uHDRIColor;
uniform vec3 uSunDirectionVector;
uniform float uHDRI_Exposure;
uniform int uUSE_HDRI;
uniform float uRoughness;
uniform int uMaterialType;

//float InvTextureWidth = 0.000244140625; // (1 / 4096 texture width)
//float InvTextureWidth = 0.00048828125;  // (1 / 2048 texture width)
//float InvTextureWidth = 0.0009765625;   // (1 / 1024 texture width)

#define INV_TEXTURE_WIDTH 0.00048828125

// the following directions pointing at the sun were found by trial and error: left here just for reference
//#define SUN_DIRECTION normalize(vec3(-0.555, 1.0, 0.205)) // use this vec3 for the symmetrical_garden_2k.hdr environment
//#define SUN_DIRECTION normalize(vec3(0.54, 1.0, -0.595)) // use this vec3 for the kiara_5_noon_2k.hdr environment

#define N_SPHERES 1
#define N_RECTANGLES 1
// #define N_BOXES 2

//-----------------------------------------------------------------------

vec3 rayOrigin, rayDirection;
// recorded intersection data:
// vec3 hitNormal, hitEmission, hitColor;
vec2 hitUV;
<<<<<<< Updated upstream
int hitType;
float hitObjectID;
=======
// int hitType;
// float hitObjectID;
>>>>>>> Stashed changes
bool hitIsModel;

struct Rectangle {
	vec3 position;
	vec3 normal;
	float radiusU;
	float radiusV;
	vec3 emission;
	vec3 color;
	int type;
};

struct Sphere {
	float radius;
	vec3 position;
	vec3 emission;
	vec3 color;
	int type;
};
// struct Box { vec3 minCorner; vec3 maxCorner; vec3 emission; vec3 color; int type; };

Sphere spheres[N_SPHERES];
Rectangle rectangles[N_RECTANGLES];
// Box boxes[N_BOXES];

#include <pathtracing_random_functions>
#include <pathtracing_calc_fresnel_reflectance>
#include <pathtracing_sphere_intersect>
// #include <pathtracing_box_intersect>
#include <pathtracing_boundingbox_intersect>
#include <pathtracing_bvhTriangle_intersect>
// #include <pathtracing_rectangle_intersect>
//#include <pathtracing_bvhDoubleSidedTriangle_intersect>
#include <pathtracing_sample_sphere_light>
//#include <pathtracing_sample_quad_light>

vec2 stackLevels[28];

//vec4 boxNodeData0 corresponds to .x = idTriangle,  .y = aabbMin.x, .z = aabbMin.y, .w = aabbMin.z
//vec4 boxNodeData1 corresponds to .x = idRightChild .y = aabbMax.x, .z = aabbMax.y, .w = aabbMax.z

void GetBoxNodeData(const in float i, inout vec4 boxNodeData0, inout vec4 boxNodeData1) {
	// each bounding box's data is encoded in 2 rgba(or xyzw) texture slots 
	float ix2 = i * 2.0;
	// (ix2 + 0.0) corresponds to .x = idTriangle,  .y = aabbMin.x, .z = aabbMin.y, .w = aabbMin.z 
	// (ix2 + 1.0) corresponds to .x = idRightChild .y = aabbMax.x, .z = aabbMax.y, .w = aabbMax.z 

	ivec2 uv0 = ivec2(mod(ix2 + 0.0, 2048.0), (ix2 + 0.0) * INV_TEXTURE_WIDTH); // data0
	ivec2 uv1 = ivec2(mod(ix2 + 1.0, 2048.0), (ix2 + 1.0) * INV_TEXTURE_WIDTH); // data1

	boxNodeData0 = texelFetch(tAABBTexture, uv0, 0);
	boxNodeData1 = texelFetch(tAABBTexture, uv1, 0);
}

mat4 m, inverse_m;
// this is the actual position of the area light rectangle - all related functions rely on this
vec3 rectanglePosition = vec3(5, 50, -36);
vec3 randPointOnRectangle;

//----------------------------------------------------------------------------------------------------------------
float RectangleIntersect(vec3 pos, vec3 normal, float radiusU, float radiusV, vec3 rayOrigin, vec3 rayDirection)
//----------------------------------------------------------------------------------------------------------------
	{
	float dt = dot(-normal, rayDirection);
	// use the following for one-sided rectangle
	//if (dt < 0.0) return INFINITY;

	float t = dot(-normal, pos - rayOrigin) / dt;
	if(t < 0.0)
		return INFINITY;

	vec3 hit = rayOrigin + rayDirection * t;
	vec3 vi = hit - pos;
	vec3 U = normalize(cross(abs(normal.y) < 0.9 ? vec3(0, 1, 0) : vec3(0, 0, 1), normal));
	vec3 V = cross(normal, U);
	return (abs(dot(U, vi)) > radiusU || abs(dot(V, vi)) > radiusV) ? INFINITY : t;
}

vec3 sampleRectangleLight(vec3 x, vec3 nl, Rectangle light, out float weight) {
	// randPointOnRectangle has already been calculated in CalculateRadiance() function,
	// so we can skip this next part

	// vec3 U = normalize(cross( abs(light.normal.y) < 0.9 ? vec3(0, 1, 0) : vec3(0, 0, 1), light.normal));
	// vec3 V = cross(light.normal, U);
	// vec3 randPointOnLight = light.position;
	// randPointOnLight += U * light.radiusU * (rng() * 2.0 - 1.0) * 0.9;
	// randPointOnLight += V * light.radiusV * (rng() * 2.0 - 1.0) * 0.9;
	// randPointOnLight = vec3(m * vec4(randPointOnLight, 1.0));

	vec3 rotatedLightNormal = normalize(vec3(m * vec4(light.normal, 0.0)));
	vec3 lightN = dot(nl, rotatedLightNormal) < 0.0 ? rotatedLightNormal : -rotatedLightNormal;

	vec3 dirToLight = randPointOnRectangle - x;
	float r2 = (light.radiusU * 2.0) * (light.radiusV * 2.0);
	float d2 = dot(dirToLight, dirToLight);
	float cos_a_max = sqrt(1.0 - clamp(r2 / d2, 0.0, 1.0));

	dirToLight = normalize(dirToLight);
	float dotNlRayDir = max(0.0, dot(nl, dirToLight));
	weight = 2.0 * (1.0 - cos_a_max) * max(0.0, -dot(dirToLight, lightN)) * dotNlRayDir;
	weight = clamp(weight, 0.0, 1.0);

	return dirToLight;
}

mat4 makeRotateX(float rot) {
	float s = sin(rot);
	float c = cos(rot);

	return mat4(1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1);
}

mat4 makeRotateY(float rot) {
	float s = sin(rot);
	float c = cos(rot);

	return mat4(c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1);
}

mat4 makeRotateZ(float rot) {
	float s = sin(rot);
	float c = cos(rot);

	return mat4(c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
}

/* Credit: Some of the equi-angular sampling code is borrowed from https://www.shadertoy.com/view/Xdf3zB posted by user 'sjb' ,
// who in turn got it from the paper 'Importance Sampling Techniques for Path Tracing in Participating Media' ,
which can be viewed at: https://docs.google.com/viewer?url=https%3A%2F%2Fwww.solidangle.com%2Fresearch%2Fegsr2012_volume.pdf */
void sampleEquiAngular(float u, float maxDistance, vec3 rOrigin, vec3 rDirection, vec3 lightPos, out float dist, out float pdf) {
	// get coord of closest point to light along (infinite) ray
	float delta = dot(lightPos - rOrigin, rDirection);

	// get distance this point is from light
	float D = distance(rOrigin + delta * rDirection, lightPos);

	// get angle of endpoints
	float thetaA = atan(0.0 - delta, D);
	float thetaB = atan(maxDistance - delta, D);

	// take sample
	float t = D * tan(mix(thetaA, thetaB, u));
	dist = delta + t;
	pdf = D / ((thetaB - thetaA) * (D * D + t * t));
}

//-----------------------------------------------------------------------------------------------------------------------------------------------
float SceneIntersect(vec3 rOrigin, vec3 rDirection, out vec3 hitNormal, out vec3 hitEmission, out vec3 hitColor, out float hitObjectID, out int hitType)
//-----------------------------------------------------------------------------------------------------------------------------------------------
	{
	vec4 currentBoxNodeData0, nodeAData0, nodeBData0, tmpNodeData0;
	vec4 currentBoxNodeData1, nodeAData1, nodeBData1, tmpNodeData1;

	vec4 vd0, vd1, vd2, vd3, vd4, vd5, vd6, vd7;

	vec3 inverseDir = 1.0 / rDirection;
	vec3 rRotatedOrigin, rRotatedDirection;

	vec2 currentStackData, stackDataA, stackDataB, tmpStackData;
	ivec2 uv0, uv1, uv2, uv3, uv4, uv5, uv6, uv7;

	float d;
	float t = INFINITY;
	float stackptr = 0.0;
	float id = 0.0;
	float tu, tv;
	float triangleID = 0.0;
	float triangleU = 0.0;
	float triangleV = 0.0;
	float triangleW = 0.0;

	int objectCount = 0;

	hitObjectID = -INFINITY;

	bool skip = false;
	bool triangleLookupNeeded = false;
	bool isRayExiting = false;
	hitIsModel = false;

	for(int i = 0; i < N_SPHERES; i++) {
		d = SphereIntersect(spheres[i].radius, spheres[i].position, rOrigin, rDirection);
		if(d < t) {
			t = d;
			hitNormal = (rOrigin + rDirection * t) - spheres[i].position;
			hitEmission = spheres[i].emission;
			hitColor = spheres[i].color;
			hitType = spheres[i].type;
			hitIsModel = false;
			hitObjectID = float(objectCount);
		}
		objectCount++;
	}

	// this part transforms the ray by the inverse of the matrix m (m holds the rectangle's transformations)
	rRotatedOrigin = vec3(inverse_m * vec4(rOrigin, 1.0));
	rRotatedDirection = vec3(inverse_m * vec4(rDirection, 0.0));

	for(int i = 0; i < N_RECTANGLES; i++) {
		// NOTE: make sure to use rRotatedOrigin and rRotatedDirection (instead of the usual ray origin and direction) in the arguments to RectangleIntersect() below
		d = RectangleIntersect(rectangles[i].position, rectangles[i].normal, rectangles[i].radiusU, rectangles[i].radiusV, rRotatedOrigin, rRotatedDirection); // <----
		if(d < t) {
			t = d;
			hitNormal = rectangles[i].normal;
			hitNormal = vec3(transpose(inverse_m) * vec4(hitNormal, 0.0));

			hitEmission = rectangles[i].emission;
			hitColor = rectangles[i].color;
			hitType = rectangles[i].type;
			hitIsModel = false;
			hitObjectID = float(objectCount);
		}
		objectCount++;
	}

	GetBoxNodeData(stackptr, currentBoxNodeData0, currentBoxNodeData1);
	currentStackData = vec2(stackptr, BoundingBoxIntersect(currentBoxNodeData0.yzw, currentBoxNodeData1.yzw, rOrigin, inverseDir));
	stackLevels[0] = currentStackData;
	skip = (currentStackData.y < t);

	while(true) {
		if(!skip) {
			// decrease pointer by 1 (0.0 is root level, 27.0 is maximum depth)
			if(--stackptr < 0.0) // went past the root level, terminate loop
				break;

			currentStackData = stackLevels[int(stackptr)];

			if(currentStackData.y >= t)
				continue;

			GetBoxNodeData(currentStackData.x, currentBoxNodeData0, currentBoxNodeData1);
		}
		skip = false; // reset skip

		if(currentBoxNodeData0.x < 0.0) // < 0.0 signifies an inner node
		{
			GetBoxNodeData(currentStackData.x + 1.0, nodeAData0, nodeAData1);
			GetBoxNodeData(currentBoxNodeData1.x, nodeBData0, nodeBData1);
			stackDataA = vec2(currentStackData.x + 1.0, BoundingBoxIntersect(nodeAData0.yzw, nodeAData1.yzw, rOrigin, inverseDir));
			stackDataB = vec2(currentBoxNodeData1.x, BoundingBoxIntersect(nodeBData0.yzw, nodeBData1.yzw, rOrigin, inverseDir));

			// first sort the branch node data so that 'a' is the smallest
			if(stackDataB.y < stackDataA.y) {
				tmpStackData = stackDataB;
				stackDataB = stackDataA;
				stackDataA = tmpStackData;

				tmpNodeData0 = nodeBData0;
				tmpNodeData1 = nodeBData1;
				nodeBData0 = nodeAData0;
				nodeBData1 = nodeAData1;
				nodeAData0 = tmpNodeData0;
				nodeAData1 = tmpNodeData1;
			} // branch 'b' now has the larger rayT value of 'a' and 'b'

			if(stackDataB.y < t) // see if branch 'b' (the larger rayT) needs to be processed
			{
				currentStackData = stackDataB;
				currentBoxNodeData0 = nodeBData0;
				currentBoxNodeData1 = nodeBData1;
				skip = true; // this will prevent the stackptr from decreasing by 1
			}
			if(stackDataA.y < t) // see if branch 'a' (the smaller rayT) needs to be processed 
			{
				if(skip) // if larger branch 'b' needed to be processed also,
					stackLevels[int(stackptr++)] = stackDataB; // cue larger branch 'b' for future round
				// also, increase pointer by 1

				currentStackData = stackDataA;
				currentBoxNodeData0 = nodeAData0;
				currentBoxNodeData1 = nodeAData1;
				skip = true; // this will prevent the stackptr from decreasing by 1
			}

			continue;
		} // end if (currentBoxNodeData0.x < 0.0) // inner node

		// else this is a leaf

		// each triangle's data is encoded in 8 rgba(or xyzw) texture slots
		id = 8.0 * currentBoxNodeData0.x;

		uv0 = ivec2(mod(id + 0.0, 2048.0), (id + 0.0) * INV_TEXTURE_WIDTH);
		uv1 = ivec2(mod(id + 1.0, 2048.0), (id + 1.0) * INV_TEXTURE_WIDTH);
		uv2 = ivec2(mod(id + 2.0, 2048.0), (id + 2.0) * INV_TEXTURE_WIDTH);

		vd0 = texelFetch(tTriangleTexture, uv0, 0);
		vd1 = texelFetch(tTriangleTexture, uv1, 0);
		vd2 = texelFetch(tTriangleTexture, uv2, 0);

		d = BVH_TriangleIntersect(vec3(vd0.xyz), vec3(vd0.w, vd1.xy), vec3(vd1.zw, vd2.x), rOrigin, rDirection, tu, tv);

		if(d < t) {
			t = d;
			triangleID = id;
			triangleU = tu;
			triangleV = tv;
			triangleLookupNeeded = true;
		}

	} // end while (true)

	if(triangleLookupNeeded) {
		//uv0 = ivec2( mod(triangleID + 0.0, 2048.0), (triangleID + 0.0) * INV_TEXTURE_WIDTH );
		//uv1 = ivec2( mod(triangleID + 1.0, 2048.0), (triangleID + 1.0) * INV_TEXTURE_WIDTH );
		uv2 = ivec2(mod(triangleID + 2.0, 2048.0), (triangleID + 2.0) * INV_TEXTURE_WIDTH);
		uv3 = ivec2(mod(triangleID + 3.0, 2048.0), (triangleID + 3.0) * INV_TEXTURE_WIDTH);
		uv4 = ivec2(mod(triangleID + 4.0, 2048.0), (triangleID + 4.0) * INV_TEXTURE_WIDTH);
		uv5 = ivec2(mod(triangleID + 5.0, 2048.0), (triangleID + 5.0) * INV_TEXTURE_WIDTH);
		//uv6 = ivec2( mod(triangleID + 6.0, 2048.0), (triangleID + 6.0) * INV_TEXTURE_WIDTH );
		//uv7 = ivec2( mod(triangleID + 7.0, 2048.0), (triangleID + 7.0) * INV_TEXTURE_WIDTH );

		//vd0 = texelFetch(tTriangleTexture, uv0, 0);
		//vd1 = texelFetch(tTriangleTexture, uv1, 0);
		vd2 = texelFetch(tTriangleTexture, uv2, 0);
		vd3 = texelFetch(tTriangleTexture, uv3, 0);
		vd4 = texelFetch(tTriangleTexture, uv4, 0);
		vd5 = texelFetch(tTriangleTexture, uv5, 0);
		//vd6 = texelFetch(tTriangleTexture, uv6, 0);
		//vd7 = texelFetch(tTriangleTexture, uv7, 0);

		// face normal for flat-shaded polygon look
		//hitNormal = normalize( cross(vec3(vd0.w, vd1.xy) - vec3(vd0.xyz), vec3(vd1.zw, vd2.x) - vec3(vd0.xyz)) );

		// interpolated normal using triangle intersection's uv's
		triangleW = 1.0 - triangleU - triangleV;
		hitNormal = triangleW * vec3(vd2.yzw) + triangleU * vec3(vd3.xyz) + triangleV * vec3(vd3.w, vd4.xy);
		hitEmission = vec3(0);
		hitColor = uMaterialColor;//vd6.yzw;
		hitUV = triangleW * vec2(vd4.zw) + triangleU * vec2(vd5.xy) + triangleV * vec2(vd5.zw);
		hitType = int(uMaterialType);//int(vd6.x);
		//hitAlbedoTextureID = -1;//int(vd7.x);
		hitIsModel = true;
		hitObjectID = float(objectCount);
	}

	return t;

} // end float SceneIntersect( out bool finalIsRayExiting )

vec3 Get_HDR_Color(vec3 rDirection) {
	vec2 sampleUV;
	//sampleUV.y = asin(clamp(rDirection.y, -1.0, 1.0)) * ONE_OVER_PI + 0.5;
	///sampleUV.x = (1.0 + atan(rDirection.x, -rDirection.z) * ONE_OVER_PI) * 0.5;
	sampleUV.x = atan(rDirection.x, -rDirection.z) * ONE_OVER_TWO_PI + 0.5;
	sampleUV.y = acos(rDirection.y) * ONE_OVER_PI;
	vec3 texColor = texture(tHDRTexture, sampleUV).rgb;

	// texColor = texData.a > 0.57 ? vec3(100) : vec3(0);
	// return texColor;
	return texColor * uHDRI_Exposure;
}

//----------------------------------------------------------------------------------------------------------------------------------------------------
vec3 CalculateRadiance(out vec3 objectNormal, out vec3 objectColor, out float objectID, out float pixelSharpness)
//----------------------------------------------------------------------------------------------------------------------------------------------------
	{
	vec3 cameraRayOrigin = rayOrigin;
	vec3 cameraRayDirection = rayDirection;
	vec3 vRayOrigin, vRayDirection;

	// recorded intersection data (from eye):
	vec3 eHitNormal, eHitEmission, eHitColor;
	float eHitObjectID;
	int eHitType = -100; // note: make sure to initialize this to a nonsense type id number!
	// recorded intersection data (from volumetric particle):
	vec3 vHitNormal, vHitEmission, vHitColor;
	float vHitObjectID;
	int vHitType = -100; // note: make sure to initialize this to a nonsense type id number!

	vec3 accumCol = vec3(0);
	vec3 mask = vec3(1);
	// vec3 checkCol0 = vec3(1);
	// vec3 checkCol1 = vec3(0.5);
	vec3 lightVec;
	vec3 particlePos;
	vec3 tdir;
	vec3 x, n, nl;

	float t, vt, camt;
	float u, xx;
	float pdf;
	float d;
	float geomTerm;
	float nc, nt, ratioIoR, Re, Tr;
	float P, RP, TP;
	float weight;
	float thickness = 0.1;
	float roughness = 0.0;
	float randomLightChoice = 0.0;

	int diffuseCount = 0;
	int previousIntersecType = -100;
	// eHitType = -100;

	bool coatTypeIntersected = false;
	bool bounceIsSpecular = true;
	bool sampleLight = false;
	bool isRayExiting = false;

	vec3 environmentCol;
	float trans;
	// fogDensity is scene-size dependent, a smaller number like 0.00001 will let most of the scene's entire fog be lit by lightsources
	// a larger number like 0.01 will only let a very small portion of the fog be lit, right around the lightsources
	float fogDensity = 0.09;
	vec3 lightPos;
	vec3 dirToLight;

	vec3 U = normalize(cross(abs(rectangles[0].normal.y) < 0.9 ? vec3(0, 1, 0) : vec3(0, 0, 1), rectangles[0].normal));
	vec3 V = cross(rectangles[0].normal, U);

	m = mat4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, rectanglePosition, 1);

	float angleX = 0.9;// angle range: 0.0 - 6.28(TWO_PI)
	float angleY = 0.3;// angle range: 0.0 - 6.28(TWO_PI)
	float angleZ = -0.6;// angle range: 0.0 - 6.28(TWO_PI)

	// multiply identity matrix m by all transformations you wish to have
	//m *= makeRotateX(angleX); // positive rotation (counter-clockwise) around +X axis
	m *= makeRotateY(angleY); // positive rotation (counter-clockwise) around +Y axis
	m *= makeRotateZ(angleZ); // positive rotation (counter-clockwise) around +Z axis

	// at this point, matrix m has been concatenated (combined), and represents all transformations
	inverse_m = inverse(m);

	for(int bounces = 0; bounces < 5; bounces++) {
		previousIntersecType = eHitType;

		randPointOnRectangle = vec3(0);
		randPointOnRectangle += U * rectangles[0].radiusU * (rng() * 2.0 - 1.0) * 0.9;
		randPointOnRectangle += V * rectangles[0].radiusV * (rng() * 2.0 - 1.0) * 0.9;

		randPointOnRectangle = vec3(m * vec4(randPointOnRectangle, 1.0));

		u = rng();

		if(u < 0.01)
			lightPos = randPointOnRectangle;
		else
			lightPos = spheres[0].position;

		t = SceneIntersect(rayOrigin, rayDirection, eHitNormal, eHitEmission, eHitColor, eHitObjectID, eHitType);

		// on first loop iteration, save intersection distance along camera ray (t) into camt variable for use below
		if(bounces == 0) {
			camt = t;
			//objectNormal = eHitNormal; // handled below
			objectColor = eHitColor;
			objectID = eHitObjectID;
		}

		// if(bounces == 0) {
		// 	objectNormal = nl;
		// 	objectColor = eHitColor;
		// 	objectID = eHitObjectID;
		// }

		// sample along intial ray from camera into the scene
		sampleEquiAngular(u, camt, cameraRayOrigin, cameraRayDirection, lightPos, xx, pdf);

		// create a particle along cameraRay and cast a shadow ray towards light (similar to Direct Lighting)
		particlePos = cameraRayOrigin + xx * cameraRayDirection;

		lightVec = lightPos - particlePos;
		d = length(lightVec);

		vRayOrigin = particlePos;
		vRayDirection = normalize(lightVec);
		roughness = hitIsModel ? uRoughness : roughness;
		vt = SceneIntersect(vRayOrigin, vRayDirection, vHitNormal, vHitEmission, vHitColor, vHitObjectID, vHitType);

		// if the particle can see the light source, apply volumetric lighting calculation
		if(vHitType == LIGHT) {
			trans = exp(-((d + xx) * fogDensity));
			geomTerm = 1.0 / (d * d);
			accumCol += vHitEmission * geomTerm * trans / pdf;
		}
		// otherwise the particle will remain in shadow - this is what produces the shafts of light vs. the volume shadows

		// useful data 
		n = normalize(eHitNormal);
		nl = dot(n, rayDirection) < 0.0 ? n : -n;
		x = rayOrigin + rayDirection * t;

		if(diffuseCount == 0) {
			objectNormal = nl;
			//objectColor = eHitColor; // handled above
			//objectID = eHitObjectID; // handled above
		}

		if(eHitType == LIGHT) {
			if(bounceIsSpecular || sampleLight) {
				trans = exp(-((camt) * fogDensity));
				accumCol += mask * eHitEmission * trans;
			}

			// reached a light, so we can exit
			break;

		} // end if (hitType == LIGHT)

		// uncomment this tiny block of code if you wish the background to be simple black (or fog color fading to black)
		// no HDRI will be shown in the background 
		if(t == INFINITY)
			break;

		// if(t == INFINITY) {
		// 	environmentCol = Get_HDR_Color(rayDirection);

		// 	// looking directly at sky
		// 	if(bounces == 0) {
		// 		pixelSharpness = 1.01;
		// 		// if(uUSE_HDRI == 0) {
		// 		// 	// accumCol = vec3(0.001,0.001,0.001);
		// 		// 	accumCol = uHDRIColor;
		// 		// } else {
		// 		// 	accumCol = environmentCol;
		// 		// }

		// 		accumCol += vec3(0.001, 0.001, 0.001);
		// 		break;
		// 	}

		// 	// sun light source location in HDRI image is sampled by a diffuse surface
		// 	// mask has already been down-weighted in this case
		// 	if(sampleLight) {
		// 		accumCol += mask * environmentCol;
		// 		break;
		// 	}

		// 	// random diffuse bounce hits sky
		// 	if(!bounceIsSpecular) {
		// 		weight = dot(rayDirection, uSunDirectionVector) < 0.98 ? 1.0 : 0.0;
		// 		accumCol += mask * environmentCol * weight;

		// 		// note: you don't need this line because the glass table and checker floor (huge sphere below) both are not in your scene setup
		// 		//if (bounces == 3) accumCol = mask * environmentCol * weight * 2.0; // checkered ground beneath glass table

		// 		break;
		// 	}

		// 	if(bounceIsSpecular) {
		// 		if(coatTypeIntersected) {
		// 			if(dot(rayDirection, uSunDirectionVector) > 0.998)
		// 				pixelSharpness = 1.01;
		// 		} else
		// 			pixelSharpness = 1.01;

		// 		if(dot(rayDirection, uSunDirectionVector) > 0.8) {
		// 			environmentCol = mix(vec3(1), environmentCol, clamp(pow(1.0 - roughness, 20.0), 0.0, 1.0));
		// 		}

		// 		accumCol += mask * environmentCol;
		// 		break;
		// 	}

		// 	// reached the HDRI sky light, so we can exit
		// 	break;

		// } // end if (t == INFINITY)

		if(bounces == 1 && diffuseCount == 0 && !coatTypeIntersected) {
			objectNormal = nl;
		}

		// if we get here and sampleLight is still true, shadow ray failed to find a light source
		if(sampleLight)
			break;

		if(eHitType == SPEC)  // Ideal SPECULAR reflection
		{
			mask *= eHitColor;

			rayDirection = randomDirectionInSpecularLobe(reflect(rayDirection, nl), 0.5);
			rayOrigin = x + nl * uEPS_intersect;
			continue;
		}

		if(eHitType == REFR)  // Ideal dielectric REFRACTION
		{
			pixelSharpness = diffuseCount == 0 ? -1.0 : pixelSharpness;

			nc = 1.0; // IOR of Air
			nt = 1.5; // IOR of common Glass
			Re = calcFresnelReflectance(rayDirection, n, nc, nt, ratioIoR);
			Tr = 1.0 - Re;
			P = 0.25 + (0.5 * Re);
			RP = Re / P;
			TP = Tr / (1.0 - P);

			if(diffuseCount == 0 && rand() < P) {
				mask *= RP;
				rayDirection = randomDirectionInSpecularLobe(reflect(rayDirection, nl), roughness);
				rayOrigin = x + nl * uEPS_intersect;
				continue;
			}

			// transmit ray through surface
			mask *= TP;

			// is ray leaving a solid object from the inside? 
			// If so, attenuate ray color with object color by how far ray has travelled through the medium
			if(isRayExiting || (distance(n, nl) > 0.1)) {
				isRayExiting = false;
				mask *= exp(log(eHitColor) * thickness * t);
			} else
				mask *= eHitColor;

			tdir = refract(rayDirection, nl, ratioIoR);
			rayDirection = randomDirectionInSpecularLobe(tdir, 0.5 * 0.5);
			rayOrigin = x - nl * uEPS_intersect;

			continue;

		} // end if (hitType == REFR)

		if(eHitType == COAT)  // Diffuse object underneath with ClearCoat on top (like car, or shiny pool ball)
		{
			coatTypeIntersected = true;

			nc = 1.0; // IOR of Air
			nt = 1.5; // IOR of Clear Coat
			Re = calcFresnelReflectance(rayDirection, nl, nc, nt, ratioIoR);
			Tr = 1.0 - Re;
			P = 0.25 + (0.5 * Re);
			RP = Re / P;
			TP = Tr / (1.0 - P);

			if(diffuseCount == 0 && rand() < P) {
				mask *= RP;
				rayDirection = randomDirectionInSpecularLobe(reflect(rayDirection, nl), 0.5);
				rayOrigin = x + nl * uEPS_intersect;
				continue;
			}

			diffuseCount++;

			mask *= eHitColor;
			mask *= TP;

			bounceIsSpecular = false;

			if(diffuseCount == 1 && rand() < 0.5) {
				mask *= 2.0;
				// choose random Diffuse sample vector
				rayDirection = randomCosWeightedDirectionInHemisphere(nl);
				rayOrigin = x + nl * uEPS_intersect;
				continue;
			}

			randomLightChoice = rng(); // will be in the range 0.0-1.0
			// in Monte Carlo path tracing, we can only select 1 possible 'dirToLight' path by random choice
			// in other words, we can...

			// either sample the rectangle area light near the model
			if(randomLightChoice < 0.7) {
				dirToLight = sampleRectangleLight(x, nl, rectangles[0], weight);
			}
			// or sample the small sphere area light
			else// if (randomLightChoice < 1.0)
			{
				dirToLight = sampleSphereLight(x, nl, spheres[0], weight);
			}

			// since we only selected 1 light source by random choice, but there are 3 light sources (much brighter)..
			// we must up-weight the contribution of the light that we did end up picking 
			weight *= 2.0; // 2.0 = number of light source choices (Rectangle, Sphere)
			// the following line also upweights because there was 0.5 chance that we reflect off of clearCoat, or sample diffuse surface beneath the clearCoat		 
			mask *= diffuseCount == 1 ? 2.0 : 1.0; // multiply by number of choices: 2.0 (either spec reflection, or diffuse)
			mask *= weight;

			rayDirection = dirToLight;
			rayOrigin = x + nl * uEPS_intersect;

			sampleLight = true;
			continue;

		} //end if (hitType == COAT)

	} // end for (int bounces = 0; bounces < 5; bounces++)

	return max(vec3(0), accumCol);

} // end vec3 CalculateRadiance( out vec3 objectNormal, out vec3 objectColor, out float objectID, out float pixelSharpness )

//-----------------------------------------------------------------------
void SetupScene(void)
//-----------------------------------------------------------------------
	{
	vec3 z = vec3(0);
	vec3 L1 = vec3(1.0, 1.0, 10.0) * 10.0;// Blueish light 
	vec3 L2 = vec3(10.0, 1.0, 10.0) * 10.0;// Pinkish light 
	// spheres[0] = Sphere(  4000.0, vec3(0, -4000, 0),  z, vec3(0.4,0.4,0.4), CHECK);//Checkered Floor
	// spheres[1] = Sphere(     6.0, vec3(55, 36, -45),  z,         vec3(0.9),  SPEC);//small mirror ball
	// spheres[2] = Sphere(     6.0, vec3(55, 24, -45),  z, vec3(0.5,1.0,1.0),  REFR);//small glass ball
	// spheres[3] = Sphere(     6.0, vec3(60, 24, -30),  z,         vec3(1.0),  COAT);//small plastic ball
	// spheres[0]  = Sphere(150.0, vec3(-400, 900, 200), L1, z, 0.0, SPEC);//spherical white Light1 
	spheres[0] = Sphere(6.0, vec3(2, 33, -55), L2, z, LIGHT);//small sphere light
	rectangles[0] = Rectangle(vec3(0), vec3(0, 0, 1), 0.4, 48.0, L1, z, LIGHT);// Rectangle Area Light
	// boxes[0] = Box( vec3(-20.0,11.0,-110.0), vec3(70.0,18.0,-20.0), z, vec3(0.2,0.9,0.7), REFR);//Glass Box
	// boxes[1] = Box( vec3(-14.0,13.0,-104.0), vec3(64.0,16.0,-26.0), z, vec3(0),           DIFF);//Inner Box
}

#include <pathtracing_main>