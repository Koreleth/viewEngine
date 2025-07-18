#Code by Oliver Zangenberg-Minde
# Modul: Grundlagen der Datenverarbetung
# from HS Fulda
# 18.07.2025

from objLoader import objLoader

o = objLoader("./woche5/obj/humanHead.obj")
#print("\nVertex-Positons:")
#print(o.vertices)
#print("\nFace-Vertex-Indices:")
#print(o.faceVertexIndices)
objects = []

for index, value in enumerate(o.faceVertexIndices[::3]):
    #print(f"Vertex {index}: {value}")
    # Get the vertex position
    v1 = o.faceVertexIndices[index * 3]
    v2 = o.faceVertexIndices[(index * 3) + 1]
    v3 = o.faceVertexIndices[(index * 3) + 2]
    print("v-number: ", v1, v2, v3)

    v_index1 = (v1 -1) * 3
    v_index2 = (v2 -1) * 3
    v_index3 = (v3 -1) * 3

    v_faces = [
        o.vertices[v_index1], o.vertices[v_index1 + 1], o.vertices[v_index1 + 2],
        o.vertices[v_index2], o.vertices[v_index2 + 1], o.vertices[v_index2 + 2],
        o.vertices[v_index3], o.vertices[v_index3 + 1], o.vertices[v_index3 + 2]
    ]
    objects.append(v_faces)
    raster_coordinates = []
    for index, val in enumerate(v_faces[::3]):
        start_index = index * 3

        #raster koordinaten werden erstellt
        v = [
            o.vertices[start_index],
            o.vertices[start_index + 1],
            o.vertices[start_index + 2]
        ]


    #print("Vertex positions: ", v)

print(objects)
