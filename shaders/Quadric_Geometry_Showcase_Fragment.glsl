precision highp float;
precision highp int;
precision highp sampler2D;

#include <pathtracing_uniforms_and_defines>

uniform mat4 uTorusInvMatrix;

#define N_LIGHTS 3.0
#define N_SPHERES 5
#define N_PLANES 1
#define N_DISKS 1
#define N_TRIANGLES 1
#define N_QUADS 1
#define N_BOXES 2
#define N_ELLIPSOIDS 2
#define N_PARABOLOIDS 1
#define N_HYPERBOLICPARABOLOIDS 1
#define N_HYPERBOLOIDS 1
#define N_OPENCYLINDERS 1
#define N_CAPPEDCYLINDERS 1
#define N_CONES 1
#define N_CAPSULES 1
#define N_TORII 1


//-----------------------------------------------------------------------

vec3 rayOrigin, rayDirection;
// recorded intersection data:
vec3 hitNormal, hitEmission, hitColor;
vec2 hitUV;
float hitObjectID;
int hitType = -100;


struct Sphere { float radius; vec3 position; vec3 emission; vec3 color; int type; };
struct Ellipsoid { vec3 radii; vec3 position; vec3 emission; vec3 color; int type; };
struct Paraboloid { float rad; float height; vec3 pos; vec3 emission; vec3 color; int type; };
struct HyperbolicParaboloid { float rad; float height; vec3 pos; vec3 emission; vec3 color; int type; };
struct Hyperboloid { float rad; float height; vec3 pos; vec3 emission; vec3 color; int type; };
struct OpenCylinder { float radius; float height; vec3 position; vec3 emission; vec3 color; int type; };
struct CappedCylinder { float radius; vec3 cap1pos; vec3 cap2pos; vec3 emission; vec3 color; int type; };
struct Cone { vec3 pos0; float radius0; vec3 pos1; float radius1; vec3 emission; vec3 color; int type; };
struct Capsule { vec3 pos0; float radius0; vec3 pos1; float radius1; vec3 emission; vec3 color; int type; };
struct UnitTorus { float parameterK; vec3 emission; vec3 color; int type; };
struct Disk { float radius; vec3 pos; vec3 normal; vec3 emission; vec3 color; int type; };


Sphere spheres[N_SPHERES];
Ellipsoid ellipsoids[N_ELLIPSOIDS];
Paraboloid paraboloids[N_PARABOLOIDS];
HyperbolicParaboloid hyperbolicParaboloids[N_HYPERBOLICPARABOLOIDS];
Hyperboloid hyperboloids[N_HYPERBOLOIDS];
OpenCylinder openCylinders[N_OPENCYLINDERS];
CappedCylinder cappedCylinders[N_CAPPEDCYLINDERS];
Cone cones[N_CONES];
Capsule capsules[N_CAPSULES];
UnitTorus torii[N_TORII];
Disk disks[N_DISKS];


#include <pathtracing_random_functions>

#include <pathtracing_calc_fresnel_reflectance>

#include <pathtracing_sphere_intersect>

#include <pathtracing_unit_bounding_sphere_intersect>

#include <pathtracing_ellipsoid_intersect>

#include <pathtracing_disk_intersect>

#include <pathtracing_opencylinder_intersect>

#include <pathtracing_cappedcylinder_intersect>

#include <pathtracing_cone_intersect>

#include <pathtracing_capsule_intersect>

#include <pathtracing_paraboloid_intersect>

#include <pathtracing_hyperboloid_intersect>

#include <pathtracing_hyperbolic_paraboloid_intersect>

#include <pathtracing_unit_torus_intersect>

#include <pathtracing_sample_sphere_light>



//-------------------------------------------------------------------------------------------------------------------
float SceneIntersect( out int finalIsRayExiting )
//-------------------------------------------------------------------------------------------------------------------
{
	vec3 rObjOrigin, rObjDirection;
	vec3 n;
	float d, dt;
	float t = INFINITY;
	int isRayExiting = FALSE;
	int insideSphere = FALSE;
	int objectCount = 0;
	
	hitObjectID = -INFINITY;

	
        for (int i = 0; i < N_SPHERES; i++)
        {
		d = SphereIntersect( spheres[i].radius, spheres[i].position, rayOrigin, rayDirection );
		if (d < t)
		{
			t = d;
			hitNormal = (rayOrigin + rayDirection * t) - spheres[i].position;
			hitEmission = spheres[i].emission;
			hitColor = spheres[i].color;
			hitType = spheres[i].type;
			hitObjectID = float(objectCount);
		}
		objectCount++;
        }
	
	for (int i = 0; i < N_ELLIPSOIDS; i++)
        {
		d = EllipsoidIntersect( ellipsoids[i].radii, ellipsoids[i].position, rayOrigin, rayDirection );
		if (d < t)
		{
			t = d;
			hitNormal = ((rayOrigin + rayDirection * t) - ellipsoids[i].position) / (ellipsoids[i].radii * ellipsoids[i].radii);
			hitEmission = ellipsoids[i].emission;
			hitColor = ellipsoids[i].color;
			hitType = ellipsoids[i].type;
			hitObjectID = float(objectCount);
		}
		objectCount++;
	}

	d = DiskIntersect( disks[0].radius, disks[0].pos, disks[0].normal, rayOrigin, rayDirection );
	if (d < t)
	{
		t = d;
		hitNormal = disks[0].normal;
		hitEmission = disks[0].emission;
		hitColor = disks[0].color;
		hitType = disks[0].type;
		hitObjectID = float(objectCount);
	}
	objectCount++;
	
	d = ParaboloidIntersect( paraboloids[0].rad, paraboloids[0].height, paraboloids[0].pos, rayOrigin, rayDirection, n );
	if (d < t)
	{
		t = d;
		hitNormal = n;
		hitEmission = paraboloids[0].emission;
		hitColor = paraboloids[0].color;
		hitType = paraboloids[0].type;
		hitObjectID = float(objectCount);
	}
	objectCount++;
	
	d = HyperbolicParaboloidIntersect( hyperbolicParaboloids[0].rad, hyperbolicParaboloids[0].height, hyperbolicParaboloids[0].pos, rayOrigin, rayDirection, n );
	if (d < t)
	{
		t = d;
		hitNormal = n;
		hitEmission = hyperbolicParaboloids[0].emission;
		hitColor = hyperbolicParaboloids[0].color;
		hitType = hyperbolicParaboloids[0].type;
		hitObjectID = float(objectCount);
	}
	objectCount++;
	
	d = HyperboloidIntersect( hyperboloids[0].rad, hyperboloids[0].height, hyperboloids[0].pos, rayOrigin, rayDirection, n );
	if (d < t)
	{
		t = d;
		hitNormal = n;
		hitEmission = hyperboloids[0].emission;
		hitColor = hyperboloids[0].color;
		hitType = hyperboloids[0].type;
		hitObjectID = float(objectCount);
	}
	objectCount++;
	
	
	d = OpenCylinderIntersect( openCylinders[0].position, openCylinders[0].position + vec3(0,30,30), openCylinders[0].radius, rayOrigin, rayDirection, n );
	if (d < t)
	{
		t = d;
		hitNormal = n;
		hitEmission = openCylinders[0].emission;
		hitColor = openCylinders[0].color;
		hitType = openCylinders[0].type;
		hitObjectID = float(objectCount);
	}
	objectCount++;
	
	d = CappedCylinderIntersect( cappedCylinders[0].cap1pos, cappedCylinders[0].cap2pos, cappedCylinders[0].radius, rayOrigin, rayDirection, n);
	if (d < t)
	{
		t = d;
		hitNormal = n;
		hitEmission = cappedCylinders[0].emission;
		hitColor = cappedCylinders[0].color;
		hitType = cappedCylinders[0].type;
		hitObjectID = float(objectCount);
	}
	objectCount++;
	
	d = ConeIntersect( cones[0].pos0, cones[0].radius0, cones[0].pos1, cones[0].radius1, rayOrigin, rayDirection, n );
	if (d < t)
	{
		t = d;
		hitNormal = n;
		hitEmission = cones[0].emission;
		hitColor = cones[0].color;
		hitType = cones[0].type;
		hitObjectID = float(objectCount);
	}
	objectCount++;
	
	d = CapsuleIntersect( capsules[0].pos0, capsules[0].radius0, capsules[0].pos1, capsules[0].radius1, rayOrigin, rayDirection, n );
	if (d < t)
	{
		t = d;
		hitNormal = n;
		hitEmission = capsules[0].emission;
		hitColor = capsules[0].color;
		hitType = capsules[0].type;
		hitObjectID = float(objectCount);
	}
	objectCount++;
	
	// transform ray into Torus's object space
	rObjOrigin = vec3( uTorusInvMatrix * vec4(rayOrigin, 1.0) );
	rObjDirection = vec3( uTorusInvMatrix * vec4(rayDirection, 0.0) );
	// first check that the ray hits the bounding sphere around the torus
	d = UnitBoundingSphereIntersect( rObjOrigin, rObjDirection, insideSphere );
	if (d < INFINITY)
	{	// if outside the sphere, move the ray up close to the Torus, for numerical stability
		d = insideSphere == TRUE ? 0.0 : d;
		rObjOrigin += rObjDirection * d;

		dt = d + UnitTorusIntersect( rObjOrigin, rObjDirection, torii[0].parameterK, n );
		if (dt < t)
		{
			t = dt;
			hitNormal = transpose(mat3(uTorusInvMatrix)) * n;
			hitEmission = torii[0].emission;
			hitColor = torii[0].color;
			hitType = torii[0].type;
			hitObjectID = float(objectCount);
		}
	}
        
	
	return t;
	
} // end float SceneIntersect( out int finalIsRayExiting )


//-----------------------------------------------------------------------------------------------------------------------------
vec3 CalculateRadiance( out vec3 objectNormal, out vec3 objectColor, out float objectID, out float pixelSharpness )
//-----------------------------------------------------------------------------------------------------------------------------
{
	Sphere lightChoice;

	vec3 accumCol = vec3(0);
        vec3 mask = vec3(1);
	vec3 reflectionMask = vec3(1);
	vec3 reflectionRayOrigin = vec3(0);
	vec3 reflectionRayDirection = vec3(0);
	vec3 checkCol0 = vec3(1);
	vec3 checkCol1 = vec3(0.5);
	vec3 dirToLight;
	vec3 tdir;
	vec3 x, n, nl;
        
	float t;
	float nc, nt, ratioIoR, Re, Tr;
	//float P, RP, TP;
	float weight;
	float thickness = 0.1;

	int diffuseCount = 0;
	int previousIntersecType = -100;
	hitType = -100;
	
	int coatTypeIntersected = FALSE;
	int bounceIsSpecular = TRUE;
	int sampleLight = FALSE;
	int isRayExiting;
	int willNeedReflectionRay = FALSE;


	lightChoice = spheres[int(rand() * N_LIGHTS)];

	
	for (int bounces = 0; bounces < 7; bounces++)
	{
		previousIntersecType = hitType;

		t = SceneIntersect(isRayExiting);
		
		/*
		//not used in this scene because we are inside a huge sphere - no rays can escape
		if (t == INFINITY)
		{
                        break;
		}
		*/

		// useful data 
		n = normalize(hitNormal);
                nl = dot(n, rayDirection) < 0.0 ? n : -n;
		x = rayOrigin + rayDirection * t;

		if (bounces == 0)
		{
			objectNormal = nl;
			objectColor = hitColor;
			objectID = hitObjectID;
		}
		if (bounces == 1 && diffuseCount == 0 && previousIntersecType == SPEC)
		{
			objectNormal = nl;
		}
		
		
		if (hitType == LIGHT)
		{	
			if (bounces == 0 || (bounces == 1 && previousIntersecType == SPEC))
				pixelSharpness = 1.01;

			if (diffuseCount == 0)
			{
				objectNormal = nl;
				objectColor = hitColor;
				objectID = hitObjectID;
			}

			if (bounceIsSpecular == TRUE || sampleLight == TRUE)
				accumCol += mask * hitEmission;

			if (willNeedReflectionRay == TRUE)
			{
				mask = reflectionMask;
				rayOrigin = reflectionRayOrigin;
				rayDirection = reflectionRayDirection;

				willNeedReflectionRay = FALSE;
				bounceIsSpecular = TRUE;
				sampleLight = FALSE;
				diffuseCount = 0;
				continue;
			}
			// reached a light, so we can exit
			break;
		} // end if (hitType == LIGHT)


		// if we get here and sampleLight is still TRUE, shadow ray failed to find the light source 
		// the ray hit an occluding object along its way to the light
		if (sampleLight == TRUE)
		{
			if (willNeedReflectionRay == TRUE)
			{
				mask = reflectionMask;
				rayOrigin = reflectionRayOrigin;
				rayDirection = reflectionRayDirection;

				willNeedReflectionRay = FALSE;
				bounceIsSpecular = TRUE;
				sampleLight = FALSE;
				diffuseCount = 0;
				continue;
			}

			break;
		}
			


		    
                if (hitType == DIFF || hitType == CHECK) // Ideal DIFFUSE reflection
		{
			if( hitType == CHECK )
			{
				float q = clamp( mod( dot( floor(x.xz * 0.04), vec2(1.0) ), 2.0 ) , 0.0, 1.0 );
				hitColor = checkCol0 * q + checkCol1 * (1.0 - q);	
			}
			// must update objectColor because hitColor may have changed
			if (bounces == 0 || (diffuseCount == 0 && coatTypeIntersected == FALSE && previousIntersecType == SPEC))	
				objectColor = hitColor;

			diffuseCount++;

			mask *= hitColor;

			bounceIsSpecular = FALSE;

			if (diffuseCount == 1 && rand() < 0.5)
			{
				mask *= 2.0;
				// choose random Diffuse sample vector
				rayDirection = randomCosWeightedDirectionInHemisphere(nl);
				rayOrigin = x + nl * uEPS_intersect;
				continue;
			}

			dirToLight = sampleSphereLight(x, nl, lightChoice, weight);
			mask *= diffuseCount == 1 ? 2.0 : 1.0;
			mask *= weight * N_LIGHTS;

			rayDirection = dirToLight;
			rayOrigin = x + nl * uEPS_intersect;

			sampleLight = TRUE;
			continue;
                        
		} // end if (hitType == DIFF)
		
		if (hitType == SPEC)  // Ideal SPECULAR reflection
		{
			mask *= hitColor;

			rayDirection = reflect(rayDirection, nl);
			rayOrigin = x + nl * uEPS_intersect;

			//if (diffuseCount == 1)
			//	bounceIsSpecular = TRUE; // turn on reflective mirror caustics

			continue;
		}
		
		if (hitType == REFR)  // Ideal dielectric REFRACTION
		{
			pixelSharpness = diffuseCount == 0 && coatTypeIntersected == FALSE ? -1.0 : pixelSharpness;
			
			nc = 1.0; // IOR of Air
			nt = 1.5; // IOR of common Glass
			Re = calcFresnelReflectance(rayDirection, n, nc, nt, ratioIoR);
			Tr = 1.0 - Re;

			if (bounces == 0 || (bounces == 1 && hitObjectID != objectID && bounceIsSpecular == TRUE))
			{
				reflectionMask = mask * Re;
				reflectionRayDirection = reflect(rayDirection, nl); // reflect ray from surface
				reflectionRayOrigin = x + nl * uEPS_intersect;
				willNeedReflectionRay = TRUE;
			}

			if (Re == 1.0)
			{
				mask = reflectionMask;
				rayOrigin = reflectionRayOrigin;
				rayDirection = reflectionRayDirection;

				willNeedReflectionRay = FALSE;
				bounceIsSpecular = TRUE;
				sampleLight = FALSE;
				continue;
			}

			// transmit ray through surface

			// is ray leaving a solid object from the inside? 
			// If so, attenuate ray color with object color by how far ray has travelled through the medium
			if (isRayExiting == TRUE)
			{
				isRayExiting = FALSE;
				mask *= exp(log(hitColor) * thickness * t);
			}
			else 
				mask *= hitColor;

			mask *= Tr;
			
			tdir = refract(rayDirection, nl, ratioIoR);
			rayDirection = tdir;
			rayOrigin = x - nl * uEPS_intersect;

			if (diffuseCount == 1)
				bounceIsSpecular = TRUE; // turn on refracting caustics

			continue;
			
		} // end if (hitType == REFR)
		
		if (hitType == COAT)  // Diffuse object underneath with ClearCoat on top
		{
			coatTypeIntersected = TRUE;
			
			nc = 1.0; // IOR of Air
			nt = 1.4; // IOR of Clear Coat
			Re = calcFresnelReflectance(rayDirection, nl, nc, nt, ratioIoR);
			Tr = 1.0 - Re;
			
			if (bounces == 0 || (bounces == 1 && hitObjectID != objectID && bounceIsSpecular == TRUE))
			{
				reflectionMask = mask * Re;
				reflectionRayDirection = reflect(rayDirection, nl); // reflect ray from surface
				reflectionRayOrigin = x + nl * uEPS_intersect;
				willNeedReflectionRay = TRUE;
			}

			diffuseCount++;

			if (bounces == 0)
				mask *= Tr;
			mask *= hitColor;

			bounceIsSpecular = FALSE;

			if (diffuseCount == 1 && rand() < 0.5)
			{
				mask *= 2.0;
				// choose random Diffuse sample vector
				rayDirection = randomCosWeightedDirectionInHemisphere(nl);
				rayOrigin = x + nl * uEPS_intersect;
				continue;
			}
			
			if (hitColor.r == 1.0 && rand() < 0.9) // this makes white capsule more 'white'
				dirToLight = sampleSphereLight(x, nl, spheres[0], weight);
			else
				dirToLight = sampleSphereLight(x, nl, lightChoice, weight);
			
			mask *= diffuseCount == 1 ? 2.0 : 1.0;
			mask *= weight * N_LIGHTS;
			
			rayDirection = dirToLight;
			rayOrigin = x + nl * uEPS_intersect;

			sampleLight = TRUE;
			continue;
			
		} //end if (hitType == COAT)

		
	} // end for (int bounces = 0; bounces < 6; bounces++)
	
	
	return max(vec3(0), accumCol);

} // end vec3 CalculateRadiance( out vec3 objectNormal, out vec3 objectColor, out float objectID, out float pixelSharpness )




//-----------------------------------------------------------------------
void SetupScene(void)
//-----------------------------------------------------------------------
{
	vec3 z  = vec3(0);          
	vec3 L1 = vec3(1.0, 1.0, 1.0) * 5.0;// White light
	vec3 L2 = vec3(1.0, 0.8, 0.2) * 4.0;// Yellow light
	vec3 L3 = vec3(0.1, 0.7, 1.0) * 2.0; // Blue light
		
        spheres[0] = Sphere(150.0, vec3(-400, 900, 200), L1, z, LIGHT);//spherical white Light1 
	spheres[1] = Sphere(100.0, vec3(300, 400, -300), L2, z, LIGHT);//spherical yellow Light2
	spheres[2] = Sphere( 50.0, vec3(500, 250, -100), L3, z, LIGHT);//spherical blue Light3
	
	spheres[3] = Sphere(1000.0, vec3(  0, 1000,  0), z, vec3(1.0, 1.0, 1.0), CHECK);//Checkered Floor Huge Sphere       
        spheres[4] = Sphere(  15.0, vec3( 32,   16, 30), z, vec3(1.0, 1.0, 1.0),  REFR);//Glass sphere
	
	disks[0] = Disk( 15.0, vec3(-100, 18,-10), vec3( 1.0,-1.0,0.0 ), z, vec3(0.01,0.3,0.7), DIFF);//BlueDisk Left

	ellipsoids[0] = Ellipsoid( vec3(30, 40, 16), vec3(90, 5, -30),  z, vec3(1.0, 0.765557, 0.336057), SPEC);//metallic gold ellipsoid
	ellipsoids[1] = Ellipsoid( vec3(5, 16, 28), vec3(-25, 16.5, 5), z, vec3(0.9, 0.9, 0.9), SPEC);//Mirror ellipsoid
	
	paraboloids[0] = Paraboloid( 20.0, 15.0, vec3(20, 2.5, -50), z, vec3(1.0, 0.2, 0.7), REFR);//paraboloid
	
	hyperbolicParaboloids[0] = HyperbolicParaboloid( 40.0, 40.0, vec3(20, 70, -50), z, vec3(1.0, 1.0, 1.0), CHECK);//hyperbolic paraboloid
	
	hyperboloids[0] = Hyperboloid( 4.0, 15.0, vec3(-15, 22, -100), z, vec3(0.3, 0.7, 0.5), REFR);//hyperboloid
	
	openCylinders[0] = OpenCylinder( 15.0, 30.0, vec3(-70, 7, -80), z, vec3(0.9, 0.01, 0.01), REFR);//red glass open Cylinder
	cappedCylinders[0] = CappedCylinder( 14.0, vec3(-60, 0, 20), vec3(-60, 14, 20), z, vec3(0.04, 0.04, 0.04), COAT);//dark gray capped Cylinder
	
	cones[0] = Cone( vec3(1, 20, -12), 15.0, vec3(1, 0, -12), 0.0, z, vec3(0.01, 0.1, 0.5), REFR);//blue Cone
	
	capsules[0] = Capsule( vec3(80, 13, 15), 10.0, vec3(110, 15.8, 15), 10.0, z, vec3(1.0,1.0,1.0), COAT);//white Capsule
	
	torii[0] = UnitTorus( 0.75, z, vec3(0.955008, 0.637427, 0.538163), SPEC);//copper Torus
		
}


#include <pathtracing_main>
