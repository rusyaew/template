#include <bits/stdc++.h>

#define PRAGMASs
#ifdef PRAGMAS
#pragma GCC optimize("Ofast")
#pragma GCC target("avx,avx2,fma")
#pragma GCC optimization ("unroll-loops")
#endif


using namespace std;
typedef long long ll;
typedef unsigned long long ull;
typedef long double ld;

void fastio() {
    ios_base::sync_with_stdio(0);
    cin.tie(nullptr);
    cout.tie(nullptr);
}



mt19937_64 gen(chrono::system_clock::now().time_since_epoch().count());
uniform_int_distribution <long long> dist(0,1e16);


//reverse-hash a...z [a; b)

const ll MOD = 1e9 + 7;
const ll p = 137;
vector<ll> pw;
vector<ll> h;
ll get(int l, int r) {
    ll finans = h[r + 1] % MOD - (h[l] * pw[r - l + 1]) % MOD;
    finans += 2*MOD;
    finans %= MOD;
    return finans % MOD;
}

int main() {
    string s;
    cin >> s;
    int n = s.size();

    h.resize(n + 1, 0);
    pw.resize(n + 1, 0);
    pw[0] = 1;
    for (int i = 0; i < n; i++) {
        h[i + 1] = (h[i] * p % MOD + (s[i] - 'a' + 1)) % MOD;
        pw[i + 1] = (pw[i] * p) % MOD;
    }
}

// 2d prefsums

int N, M;
bool table[305][305];
int preftrees[306][306];
void prefsums_precalc() {
    for (int x = 1; x <= N; x++) {
        for (int y = 1; y <= M; y++) {
            int val = preftrees[x - 1][y] + preftrees[x][y - 1] - preftrees[x - 1][y - 1] + table[x - 1][y - 1];
            preftrees[x][y] = val;
        }
    }
}
int amountoftrees_pref(int x1, int y1, int x2, int y2) {
    int ans = preftrees[x2 + 1][y2 + 1] - preftrees[x2 + 1][y1] - preftrees[x1][y2 + 1] + preftrees[x1][y1];
    return ans;
}


//first greater left

long long base_arr_size;
vector<long long> base_arr;
vector< pair<long long, long long> > stackk;
vector<long long> ans_array;

base_arr_size = base_arr.size();
ans_array.resize(base_arr_size);
stackk = {make_pair(-1, 1e10)};
for (long long i = 0; i < base_arr_size; i++) {
while (stackk.back().second <= base_array[i]) {
stackk.pop_back();
}
ans_array[i] = stackk.back().first;
stackk.push_back(make_pair(i, base_arr[i]));
}

// dsu

vector<int> p;
vector<int> size;
void init(int n) {
    p.resize(n);
    for (int i = 0; i < n; ++i) {
        p[i] = i;
    }
    size.resize(n, 1);
}
int GetRoot(int i) {
    if (p[i] == i) {
        return i;
    }
    return p[i] = GetRoot(p[i]);
}

void Merge(int i, int j) {
    int root_i = GetRoot(i);
    int root_j = GetRoot(j);
    if (root_i == root_j) {
        return;
    }
    if (size[root_i] < size[root_j]) {
        p[root_i] = root_j;
        size[root_j] += size[root_i];
    } else {
        p[root_j] = root_i;
        size[root_i] += size[root_j];
    }
}

// djikstra

vector<int> dijkstra(int s) {
    vector<int> d(n, inf);
    d[root] = 0;
    set< pair<int, int> > q;
    q.insert({0, s});
    while (!q.empty()) {
        int v = q.begin()->second;
        q.erase(q.begin());
        for (auto [u, w] : g[v]) {
            if (d[u] > d[v] + w) {
                q.erase({d[u], u});
                d[u] = d[v] + w;
                q.insert({d[u], u});
            }
        }
    }
    return d;
}
// sqrt decompositon

struct Block {
    vector<long long> data;
    vector<long long> srt;
    long long size = 0;
    void build() {
        size = data.size();
        srt = data;
        sort(srt.begin(), srt.end());
    }
    long long FullAmount(long long X) {
        if (size == 0) {
            return 0;
        }

        auto ubnd = upper_bound(srt.begin(), srt.end(), X);
        long long ubnd_index = ubnd - srt.begin();
        return ubnd_index;
    }
    long long PartAmount(long long L, long long R, long long X) {
        long long ans = 0;
        for (long long i = L; i <= R; i++) {
            ans += (data[i] <= X);
        }
        return ans;
    }
    void DeleteElement(long long I) {
        auto el_it = data.begin() + I;
        srt.erase(lower_bound(srt.begin(), srt.end(), *el_it));
        data.erase(el_it);
        size--;
    }
    void InsertElement(long long I, long long X) {
        auto el_it = data.begin() + I;
        srt.insert(lower_bound(srt.begin(), srt.end(), X), X);
        data.insert(el_it, X);
        size++;
    }
};

struct SQRTDecomposition {
    vector<Block> Blocks;
    long long block_size = 20;
    long long cnt = 0;

    SQRTDecomposition(vector<long long> base, long long block_len) {
        long long full_size = base.size();
        block_size = block_len;
        if (full_size == 0) {
            Blocks.push_back(Block());
            Blocks[0].build();
            return;
        }
        for (long long i = 0; i < full_size; i++) {
            long long block_id = i / block_size;
            if (block_id == cnt) {
                Blocks.push_back(Block());
                cnt++;
            }
            Blocks[block_id].data.push_back(base[i]);
        }
        for (long long i = 0; i < cnt; i++) {
            Blocks[i].build();
        }
    }
    long long AmountOnRange(long long L, long long R, long long x) {
        long long ans = 0;
        long long PosBlockL = 0;
        long long PosBlockLBeginning = 0;
        while (L - Blocks[PosBlockL].size >= 0) {
            L -= Blocks[PosBlockL].size;
            PosBlockL++;
        }
        long long PosBlockR = 0;
        long long PosBlockRBeginning = 0;
        while (R - Blocks[PosBlockR].size >= 0) {
            R -= Blocks[PosBlockR].size;
            PosBlockR++;
        }
        if (PosBlockL == PosBlockR) {
            return Blocks[PosBlockL].PartAmount(L, R, x);
        }
        if (L != 0) {
            ans += Blocks[PosBlockL].PartAmount(L, Blocks[PosBlockL].size - 1, x);
            PosBlockL++;
        }
        if (R != Blocks[PosBlockR].size - 1) {
            ans += Blocks[PosBlockR].PartAmount(0, R, x);

            PosBlockR--;
        }
        for (int i = PosBlockL; i <= PosBlockR; i++) {
            ans += Blocks[i].FullAmount(x);
        }
        return ans;
    }

    void DeleteElement(long long I) {
        long long PosBlockL = 0;
        while (I - Blocks[PosBlockL].size >= 0) {
            I -= Blocks[PosBlockL].size;
            PosBlockL++;
        }
        Blocks[PosBlockL].DeleteElement(I);
    }

    void InsertElement(long long I, long long X) {
        long long PosBlockL = 0;
        while (I - Blocks[PosBlockL].size >= 0) {
            I -= Blocks[PosBlockL].size;
            PosBlockL++;

            //printf("I = %d, PosBlockL = %d \n", I, PosBlockL);

            if (I == 0 && PosBlockL == Blocks.size()) {
                PosBlockL = Blocks.size() - 1;
                I = Blocks[PosBlockL].size;
                break;
            }
        }
        Blocks[PosBlockL].InsertElement(I, X);
    }
};

//segtree mass-sum

const int maxn = 1e6 + 1;
ll t[4*maxn];
ll add[4*maxn];
vector<ll> a;
void build(int v, int l, int r) {
    if (l + 1 == r) {
        t[v] = a[l];
        return;
    }

    int m = (l + r ) / 2;
    build(v<<1, l, m); build(v<<1|1, m, r);
    t[v] = t[v<<1] + t[v<<1|1];
}
void push(int v) {
    add[v<<1] += add[v];
    add[v<<1|1] += add[v];
    add[v] = 0;
}
void rangeadd(int v, int tl, int tr, int l, int r, int x) {
    if (tr <= l || tl >= r) {
        return;
    } else if (l <= tl && tr <= r) {
        add[v] += x;
        return;
    }

    push(v);
    int tm = (tl + tr) * .5;
    rangeadd(v<<1, tl, tm, l, r, x);
    rangeadd(v<<1|1, tm, tr, l, r, x);
    t[v] = t[v<<1] + t[v<<1|1] + add[v<<1] * (tm - tl) + add[v<<1|1] * (tr - tm);
}
ll rangesum(int v, int tl, int tr, int l, int r) {
    if (tr <= l || tl >= r) {
        return 0;
    } else if (l <= tl && tr <= r) {
        return t[v] + add[v] * (tr - tl);
    }

    push(v);
    int tm = (tl + tr) * .5;
    ll ans = 0;
    ans += rangesum(v<<1, tl, tm, l, r);
    ans += rangesum(v<<1|1, tm, tr, l, r);
    t[v] = t[v<<1] + t[v<<1|1] + add[v<<1] * (tm - tl) + add[v<<1|1] * (tr - tm);
    return ans;
}

//rotate matrix NxN 90 deg clockwise

for (int i=0;i<n/2;i++)
{
for (int j=i;j<n-i-1;j++)
{
int temp=arr[i][j];
arr[i][j]=arr[n-1-j][i];
arr[n-1-j][i]=arr[n-1-i][n-1-j];
arr[n-1-i][n-1-j]=arr[j][n-1-i];
arr[j][n-1-i]=temp;
}
}


//rotate matrix NxN 90 deg anti-clockwise

for(int i=0;i<n/2;i++)
{
for(int j=i;j<n-i-1;j++)
{
// Swapping elements after each iteration in Anticlockwise direction
int temp=arr[i][j];
arr[i][j]=arr[j][n-i-1];
arr[j][n-i-1]=arr[n-i-1][n-j-1];
arr[n-i-1][n-j-1]=arr[n-j-1][i];
arr[n-j-1][i]=temp;
}
}


// modulo division

long long inv(long long a, long long b){
    return 1<a ? b - inv(b % a, a) * b/a : 1;
}

long long a_divided_by_b_modulo_p(int a, int b, int p) {
    // given that p - prime, a, b - coprime
    long long aa = a % p;
    long long b_inverse = (inv(b, p)) % p;
    long long result = (aa * b_inverse) % MOD;
    return result;
}


// fast factorization

inline ll f(ll x, ll n) { return (__int128_t) (x + 1) * (x + 1) % n; }

ll find_divisor(ll n, ll seed = 1) {
    ll x = seed, y = seed;
    ll d = 1;
    while (d == 1 || d == n) {
        y = f(y);
        x = f(f(x));
        d = gcd(abs(x - y), n);
    }
    return d;
}


// fast eratosphen

const int n = 1e6;

int d[n + 1];
vector<int> p;

for (int k = 2; k <= n; k++) {
if (p[k] == 0) {
d[k] = k;
p.push_back(k);
}
for (int x : p) {
if (x > d[k] || x * d[k] > n)
break;
d[k * x] = x;
}
}


// extended gcd

int gcd (int a, int b, int & x, int & y) {
    if (a == 0) {
        x = 0; y = 1;
        return b;
    }
    int x1, y1;
    int d = gcd (b%a, a, x1, y1);
    x = y1 - (b / a) * x1;
    y = x1;
    return d;
}


// datetime

datetime.date(year, month, day)
datetime.time(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
datetime.timedelta
        datetime.tzinfo
        datetime.datetime(year, month, day, hour=0, minute=0, second=0, microsecond=0, tzinfo=None)


// LCA bin

bool a (int u, int v) {
    return tin[u] <= tin[v] && tin[v] < tout[u];
}

int up[maxn][logn];
void dfs (int v) {
    for (int l = 1; l < logn; l++)
        up[v][l] = up[up[v][l-1]][l-1];
    tin[v] = t++;
    for (int u : g[v]) {
        up[u][0] = v;
        dfs(u);
    }
    tout[v] = t;
}

int lca (int v, int u) {
    if (a(v, u)) return v;
    if (a(u, v)) return u;
    for (int l = logn-1; l >= 0; l--)
        if (!ancestor(up[v][l], u))
            v = up[v][l];
    return up[v][0];
}


// Kraskal min spanning tree

struct Edge {
    int from, to, weight;
};

vector<Edge> edges;

sort(edges.begin(), edges.end(), [](Edge a, Edge b) {
    return a.weight < b.weight;
});

for (auto [a, b, w] : edges) {
    if (p(a) != p(b)) {
        unite(a, b);
    }
}


// indexed set

#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
using namespace std;
typedef tree<int, null_type, less<int>, rb_tree_tag, tree_order_statistics_node_update> indexed_set;

indexed_set s;
s.insert(1);
auto x = s.find_by_order(2); // что находится на такой позиции в отсорт массиве
s.order_of_key(7) // на какой позиции в отсортированном массиве находится


// maximum subarray 2d

const int MAX_N = 1e5 + 1;
const int MOD = 1e9 + 7;
const int INF = 1e9;
const ll LINF = 1e18;



void solve() {
    int n; cin >> n;
    ll ps[n + 1][n + 1] = {};
    // Creating a prefix sum table (horizontally)
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            cin >> ps[i][j];
            ps[i][j] += ps[i][j - 1];
        }
    }

    // 2D max sum subarray
    ll ans = 0;
    for (int l = 1; l <= n; l++) { // looping through all pairs of columns, running 1D maxsum on a column
        for (int r = l; r <= n; r++) {
            ll cur = 0;
            for (int i = 1; i <= n; i++) {
                ll x = ps[i][r] - ps[i][l - 1];
                cur = max(0ll, cur + x);
                ans = max(ans, cur);
            }
        }
    }
    cout << ans << "\n";
}

// RMQ LCA Sparse Table
const int MAX_N = 2e5 + 1;
const int MOD = 1e9 + 7;
const int INF = 1e9;
const ll LINF = 1e18;

const int MAX_L = 30;

int n, q, idx;
int dep[2 * MAX_N], euler[2 * MAX_N], first[MAX_N], lg[2 * MAX_N];
ar<int,2> dp[2 * MAX_N][MAX_L]; // need to store both the indices and the min value
vector<int> adj[MAX_N];

void dfs(int u, int p = 0, int h = 0) {
    euler[++idx] = u;
    dep[idx] = h;
    first[u] = idx;
    for (int v : adj[u]) {
        if (v != p) {
            dfs(v, u, h + 1);
            euler[++idx] = u;
            dep[idx] = h;
        }
    }
}

void build_lg_table() {
    lg[1] = 0;
    for (int i = 2; i <= 2 * n; i++)
        lg[i] = lg[i / 2] + 1;
}

void build_sparse_table() {
    for (int i = 1; i <= 2 * n; i++)
        dp[i][0] = {dep[i], euler[i]};
    for (int j = 1; j < MAX_L; j++)
        for (int i = 1; i + (1 << j) <= 2 * (n + 1); i++)
            dp[i][j] = min(dp[i][j - 1], dp[i + (1 << (j - 1))][j - 1]);
}

int min_query(int l, int r) {
    int k = lg[r - l + 1];
    return min(dp[l][k], dp[r - (1 << k) + 1][k])[1]; // return the index with min value
}

int lca(int u, int v) {
    int l = first[u], r = first[v];
    if (l > r) swap(l, r);
    return min_query(l, r);
}

void solve() {
    cin >> n >> q;
    for (int v = 2; v <= n; v++) {
        int u; cin >> u;
        adj[u].push_back(v);
    }
    dfs(1);
    build_lg_table();
    build_sparse_table();
    while (q--) {
        int a, b; cin >> a >> b;
        cout << lca(a, b) << "\n";
    }
    cout << "\n";
}

int main() {
    ios_base::sync_with_stdio(0);
    cin.tie(0); cout.tie(0);
    // freopen("input.txt", "r", stdin);
    // freopen("output.txt", "w", stdout);

    int tc; tc = 1;
    for (int t = 1; t <= tc; t++) {
        // cout << "Case #" << t  << ": ";
        solve();
    }
}
//longest incr subsec
void solve() {
    int n; cin >> n;
    vector<int> dp;
    for (int i = 0; i < n; i++) {
        int x; cin >> x;
        auto it = lower_bound(dp.begin(), dp.end(), x);
        if (it == dp.end()) dp.push_back(x);
        else *it = x;
    }
    cout << dp.size() << "\n";
}

// qucick nck

const int MAX_N = 1e5 + 5;
const ll MOD = 1e9 + 7;
const ll INF = 1e9;

ll qexp(ll a, ll b, ll m) {
    ll res = 1;
    while (b) {
        if (b % 2) res = res * a % m;
        a = a * a % m;
        b /= 2;
    }
    return res;
}

vector<ll> fact, invf;

void precompute(int n) {
    fact.assign(n + 1, 1);
    for (int i = 1; i <= n; i++) fact[i] = fact[i - 1] * i % MOD;
    invf.assign(n + 1, 1);
    invf[n] = qexp(fact[n], MOD - 2, MOD);
    for (int i = n - 1; i > 0; i--) invf[i] = invf[i + 1] * (i + 1) % MOD;
}

ll nCk(int n, int k) {
    if (k < 0 || k > n) return 0;
    return fact[n] * invf[k] % MOD * invf[n - k] % MOD;
    // return fact[n] * qexp(fact[k], MOD - 2, MOD) % MOD * qexp(fact[n - k], MOD - 2, MOD) % MOD;
}

// A trick to calculate large factorial without overflowing is to take log at every step when precompute and take exponential when calculating
// Don't need invf[] now because it is the same as negative log of fact
vector<double> log_fact;
void precompute_log(int n) {
    log_fact.assign(n + 1, 0.0);
    log_fact[0] = 0.0;
    for (int i = 1; i <= n; i++) log_fact[i] = log_fact[i - 1] + log(i);
}

ll log_nCk(int n, int k) {
    if (k < 0 || k > n) return 0;
    return exp(log_fact[n] - log_fact[n - k] - log_fact[k]);
}

void solve() {
    int n, k; cin >> n >> k;
    cout << nCk(n, k) << "\n";
}

int main() {
    ios_base::sync_with_stdio(0);
    cin.tie(0); cout.tie(0);
    precompute(1e6);
    int tc = 1;
    cin >> tc;
    for (int t = 1; t <= tc; t++) {
        // cout << "Case #" << t << ": ";
        solve();
    }
}
