#pragma once

#include "scene.cuh"
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath>

inline std::vector<std::string> splitStr(const std::string& str) {
    std::vector<std::string> result;
    std::string cleaned = str;

    cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), '\r'), cleaned.end());
    cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), '\n'), cleaned.end());

    std::istringstream iss(cleaned);
    std::string token;
    while (iss >> token)
        result.push_back(token);
    return result;
}

class Mesh {
public:
    std::vector<Vec3> vertices;
    std::vector<Vec3> normals;
    std::vector<int>  faceVerts;
    std::vector<int>  faceNorms;   // 3 indices per triangle -> normal (empty means flat)
    Vec3 color;
    Vec3 emis;
    ReflType refl;

    Mesh(const Vec3& col = Vec3(0.7f), const Vec3& e = Vec3(0.f), ReflType r = DIFF)
        : color(col), emis(e), refl(r) {}

    int loadOBJ(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Mesh::loadOBJ: failed to open " << filename << std::endl;
            return -1;
        }

        vertices.clear();
        normals.clear();
        faceVerts.clear();
        faceNorms.clear();

        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string type;
            iss >> type;

            if (type.empty() || type[0] == '#')
                continue;

            if (type == "v") {
                float x, y, z;
                iss >> x >> y >> z;
                vertices.emplace_back(x, y, z);
            }
            else if (type == "vn") {
                float x, y, z;
                iss >> x >> y >> z;
                normals.emplace_back(x, y, z);
            }
            else if (type == "f") {
                auto tokens = splitStr(line);
                // triangulate fan
                for (int i = 3; i < (int)tokens.size(); ++i) {
                    parseFaceVertex(tokens[1]);
                    parseFaceVertex(tokens[i - 1]);
                    parseFaceVertex(tokens[i]);
                }
            }
        }
        file.close();
        return 0;
    }

    // convert mesh to flat Shape array so it would be ready for gpu
    void toShapes(std::vector<Shape>& out) const {
        int triCount = (int)faceVerts.size() / 3;
        bool hasNormals = !faceNorms.empty();

        for (int i = 0; i < triCount; ++i) {
            Shape t{};
            t.type = SHAPE_TRIANGLE;
            t.color = color;
            t.emis = emis;
            t.refl = refl;

            int i0 = faceVerts[i * 3 + 0];
            int i1 = faceVerts[i * 3 + 1];
            int i2 = faceVerts[i * 3 + 2];

            t.tri.v0 = vertices[i0];
            t.tri.v1 = vertices[i1];
            t.tri.v2 = vertices[i2];

            t.tri.normal = (t.tri.v1 - t.tri.v0).cross(t.tri.v2 - t.tri.v0).norm();

            if (hasNormals) {
                int n0 = faceNorms[i * 3 + 0];
                int n1 = faceNorms[i * 3 + 1];
                int n2 = faceNorms[i * 3 + 2];
                t.tri.vn0 = normals[n0];
                t.tri.vn1 = normals[n1];
                t.tri.vn2 = normals[n2];
            } else {
                t.tri.vn0 = t.tri.vn1 = t.tri.vn2 = Vec3(0.f);
            }

            out.push_back(t);
        }
    }

    void translate(const Vec3& delta) {
        for (auto& v : vertices) v = v + delta;
    }

    void scale(float sx, float sy, float sz) {
        for (auto& v : vertices) {
            v.x *= sx; v.y *= sy; v.z *= sz;
        }

        if (sx != sy || sy != sz) {
            float isx = 1.f / sx, isy = 1.f / sy, isz = 1.f / sz;
            for (auto& n : normals) {
                n.x *= isx; n.y *= isy; n.z *= isz;
                float len = n.length();
                if (len > 0.f) n = n / len;
            }
        }
    }

    void scale(float s) { scale(s, s, s); }

    void rotateY(float angleDeg) {
        float a = angleDeg * (float(M_PI) / 180.f);
        float ca = cosf(a), sa = sinf(a);

        auto rot = [ca, sa](Vec3& v) {
            float x = v.x * ca + v.z * sa;
            float z = -v.x * sa + v.z * ca;
            v.x = x; v.z = z;
        };

        for (auto& v : vertices) rot(v);
        for (auto& n : normals) rot(n);
    }

    void rotateX(float angleDeg) {
        float a = angleDeg * (float(M_PI) / 180.f);
        float ca = cosf(a), sa = sinf(a);

        auto rot = [ca, sa](Vec3& v) {
            float y = v.y * ca - v.z * sa;
            float z = v.y * sa + v.z * ca;
            v.y = y; v.z = z;
        };

        for (auto& v : vertices) rot(v);
        for (auto& n : normals) rot(n);
    }

    void rotateZ(float angleDeg) {
        float a = angleDeg * (float(M_PI) / 180.f);
        float ca = cosf(a), sa = sinf(a);

        auto rot = [ca, sa](Vec3& v) {
            float x = v.x * ca - v.y * sa;
            float y = v.x * sa + v.y * ca;
            v.x = x; v.y = y;
        };

        for (auto& v : vertices) rot(v);
        for (auto& n : normals) rot(n);
    }

    Vec3 centroid() const {
        if (vertices.empty()) return Vec3(0.f);
        Vec3 sum(0.f);
        for (const auto& v : vertices) sum += v;
        return sum * (1.f / (float)vertices.size());
    }

    void rotateAroundCenter(float angleDeg, int axis) {
        Vec3 c = centroid();
        translate(-c);
        if (axis == 0) rotateX(angleDeg);
        else if (axis == 1) rotateY(angleDeg);
        else rotateZ(angleDeg);
        translate(c);
    }

    void scaleAroundCenter(float sx, float sy, float sz) {
        Vec3 c = centroid();
        translate(-c);
        scale(sx, sy, sz);
        translate(c);
    }

    void scaleAroundCenter(float s) { scaleAroundCenter(s, s, s); }

private:
    void parseFaceVertex(const std::string& token) {
        int vi = 0, ni = -1;

        std::istringstream iss(token);
        iss >> vi;
        faceVerts.push_back(vi > 0 ? vi - 1 : (int)vertices.size() + vi);

        char ch = (char)iss.peek();
        if (ch == '/') {
            iss.ignore();
            ch = (char)iss.peek();
            if (ch == '/') {
                iss.ignore();
                iss >> ni;
            }
            else if (isdigit(ch) || ch == '-') {
                int tex = 0;
                iss >> tex;
                ch = (char)iss.peek();
                if (ch == '/') {
                    iss.ignore();
                    iss >> ni;
                }
            }
        }

        faceNorms.push_back(ni > 0 ? ni - 1 : (ni < 0 ? -1 : (int)normals.size() + ni));
    }
};

inline int loadMeshToShapes(
    std::vector<Shape>& out,
    const std::string& filename,
    const Vec3& color = Vec3(0.7f),
    const Vec3& emis = Vec3(0.f),
    ReflType refl = DIFF
) {
    Mesh mesh(color, emis, refl);
    if (mesh.loadOBJ(filename) != 0)
        return -1;
    mesh.toShapes(out);
    return 0;
}
