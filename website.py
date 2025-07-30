import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx
import plotly.graph_objects as go

st.set_page_config(layout="wide")import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("📊 Visualisasi Aturan Asosiasi Modul")

with st.expander("📌 Format CSV yang Diperlukan"):
    st.markdown("""
    File CSV harus memiliki **satu kolom bernama `nama_modul`**.

    - Setiap baris berisi daftar modul dalam satu file/proyek
    - Modul dipisahkan dengan koma
    
    **Contoh isi CSV:**
    ```
    nama_modul
    os,sys,logging
    pandas,numpy,matplotlib
    requests,json
    ```
    """)

uploaded_file = st.file_uploader("📁 Unggah file CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "nama_modul" not in df.columns:
        st.error("❌ Kolom `nama_modul` tidak ditemukan.")
        st.stop()

    support = st.selectbox("Pilih minimum support", [0.1, 0.05, 0.02, 0.01, 0.005])
    confidence = st.selectbox("Pilih minimum confidence", [0.9, 0.7, 0.5, 0.3, 0.1])

    df["nama_modul"] = df["nama_modul"].astype(str)
    transactions = df["nama_modul"].apply(lambda x: [mod.strip() for mod in x.split(',')]).tolist()

    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)

    try:
        freq_items = apriori(df_encoded, min_support=support, use_colnames=True)
    except MemoryError:
        st.error("⚠️ Terjadi error memori. Coba naikkan nilai minimum support atau kurangi ukuran data.")
        st.stop()

    if freq_items.empty:
        st.warning("Tidak ada itemset yang memenuhi minimum support.")
        st.stop()

    rules = association_rules(freq_items, metric="confidence", min_threshold=confidence)
    rules = rules[
        (rules['antecedents'].apply(lambda x: len(x) == 1)) &
        (rules['consequents'].apply(lambda x: len(x) == 1))
    ]

    if rules.empty:
        st.warning("Tidak ditemukan aturan asosiasi.")
    else:
        rules['antecedents'] = rules['antecedents'].apply(lambda x: next(iter(x)))
        rules['consequents'] = rules['consequents'].apply(lambda x: next(iter(x)))

        rules.rename(columns={
            'antecedents': 'Modul A',
            'consequents': 'Modul B',
            'support': 'Support',
            'confidence': 'Confidence'
        }, inplace=True)

        rules.sort_values(by=['Support', 'Confidence'], ascending=[False, False], inplace=True)

        rules_display = rules.copy()
        rules_display["Support"] = rules_display["Support"].apply(lambda x: f"{x:.3f}")
        rules_display["Confidence"] = rules_display["Confidence"].apply(lambda x: f"{x:.3f}")

        st.subheader("📄 Aturan Asosiasi Modul")
        st.dataframe(rules_display[["Modul A", "Modul B", "Support", "Confidence"]])

        # Visualisasi graph
        G = nx.DiGraph()
        for _, row in rules.iterrows():
            G.add_edge(row['Modul A'], row['Modul B'], confidence=row['Confidence'])

        if G.number_of_edges() == 0:
            st.info("Tidak ada hubungan yang bisa divisualisasikan.")
        else:
            pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
            node_x, node_y = [], []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)

            edge_x, edge_y = [], []
            edge_hover_x, edge_hover_y, edge_hover_text = [], [], []

            for edge in G.edges(data=True):
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                edge_hover_x.append(mx)
                edge_hover_y.append(my)

                conf = edge[2]['confidence']
                hovertext = f"{edge[0]} → {edge[1]}<br>Confidence: {conf:.3f}"

                reverse = rules[
                    (rules['Modul A'] == edge[1]) & (rules['Modul B'] == edge[0])
                ]
                if not reverse.empty:
                    rc = reverse.iloc[0]['Confidence']
                    hovertext += f"<br><br>{edge[1]} → {edge[0]}<br>Confidence: {rc:.3f}"

                edge_hover_text.append(hovertext)

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y, line=dict(width=1, color='#888'),
                hoverinfo='none', mode='lines'
            )

            edge_hover_trace = go.Scatter(
                x=edge_hover_x, y=edge_hover_y,
                mode='markers', hoverinfo='text',
                marker=dict(size=10, color='rgba(0,0,0,0)'),
                hovertext=edge_hover_text,
                showlegend=False
            )

            node_color = [
                rules[rules['Modul A'] == node]['Confidence'].sum() if node in rules['Modul A'].values else 0
                for node in G.nodes()
            ]

            node_hover_text = []
            for node in G.nodes():
                total_conf = rules[rules['Modul A'] == node]['Confidence'].sum()
                node_hover_text.append(f"Modul: {node}<br>Total Confidence: {total_conf:.3f}")

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=[str(n) for n in G.nodes()],
                textposition="bottom center",
                hoverinfo='text',
                marker=dict(
                    showscale=True,
                    colorscale='YlGnBu',
                    color=node_color,
                    size=20,
                    colorbar=dict(title=dict(text='Total Confidence', side='right')),
                    line_width=2
                ),
                hovertext=node_hover_text
            )

            fig = go.Figure(
                data=[edge_trace, edge_hover_trace, node_trace],
                layout=go.Layout(
                    title="🔗 Visualisasi Hubungan Asosiasi Modul",
                    title_x=0.4,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                )
            )

            st.plotly_chart(fig, use_container_width=True)

st.title("Module Association Rule Visualization")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'nama_modul' not in df.columns:
        st.error("The 'nama_modul' column was not found in the uploaded file.")
    else:
        support = st.selectbox("Select minimum support", [0.1, 0.05, 0.02, 0.01, 0.005])
        confidence = st.selectbox("Select minimum confidence", [0.9, 0.7, 0.5, 0.3, 0.1])

        df["nama_modul"] = df["nama_modul"].astype(str)
        transactions = df["nama_modul"].apply(lambda x: [mod.strip() for mod in x.split(',')]).tolist()

        te = TransactionEncoder()
        te_array = te.fit(transactions).transform(transactions, sparse=True)
        df_encoded = pd.DataFrame.sparse.from_spmatrix(te_array, columns=te.columns_)

        try:
            freq_items = apriori(df_encoded, min_support=support, use_colnames=True)
        except MemoryError:
            st.error("Memory error: Try increasing the minimum support or reducing the data size.")
            st.stop()

        if freq_items.empty:
            st.warning("No itemsets meet the minimum support. Try lowering the support value.")
            st.stop()
        
        rules = association_rules(freq_items, metric="confidence", min_threshold=confidence)

        # Filter rules with one item in both antecedent and consequent
        rules = rules[
            (rules['antecedents'].apply(lambda x: len(x) == 1)) &
            (rules['consequents'].apply(lambda x: len(x) == 1))
        ]

        if rules.empty:
            st.warning("No association rules were found.")
        else:
            # Convert frozensets to string
            rules['antecedents'] = rules['antecedents'].apply(lambda x: next(iter(x)))
            rules['consequents'] = rules['consequents'].apply(lambda x: next(iter(x)))

            rules.rename(columns={
                'antecedents': 'Module A',
                'consequents': 'Module B',
                'support': 'Support',
                'confidence': 'Confidence'
            }, inplace=True)

            # Sort by support then confidence
            rules.sort_values(by=['Support', 'Confidence'], ascending=[False, False], inplace=True)

            rules_display = rules.copy()
            rules_display["Support"] = rules_display["Support"].apply(lambda x: f"{x:.3f}")
            rules_display["Confidence"] = rules_display["Confidence"].apply(lambda x: f"{x:.3f}")

            st.subheader("Association Rules")
            st.dataframe(rules_display[["Module A", "Module B", "Support", "Confidence"]])

            # Build association graph
            G = nx.DiGraph()
            for _, row in rules.iterrows():
                G.add_edge(row['Module A'], row['Module B'],
                           confidence=row['Confidence'])

            if G.number_of_edges() == 0:
                st.info("No results to display in the visualization.")
            else:
                pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

                node_x, node_y = [], []
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)

                edge_x, edge_y = [], []
                edge_hover_x, edge_hover_y, edge_hover_text = [], [], []

                for edge in G.edges(data=True):
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x += [x0, x1, None]
                    edge_y += [y0, y1, None]

                    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                    edge_hover_x.append(mx)
                    edge_hover_y.append(my)

                    conf = edge[2]['confidence']
                    hovertext = (
                        f"{edge[0]} → {edge[1]}<br>"
                        f"Confidence: {conf:.3f}"
                    )

                    reverse = rules[
                        (rules['Module A'] == edge[1]) & (rules['Module B'] == edge[0])
                    ]
                    if not reverse.empty:
                        rc = reverse.iloc[0]['Confidence']
                        hovertext += (
                            f"<br><br>{edge[1]} → {edge[0]}<br>"
                            f"Confidence: {rc:.3f}"
                        )

                    edge_hover_text.append(hovertext)

                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y, line=dict(width=1, color='#888'),
                    hoverinfo='none', mode='lines'
                )

                edge_hover_trace = go.Scatter(
                    x=edge_hover_x, y=edge_hover_y,
                    mode='markers', hoverinfo='text',
                    marker=dict(size=10, color='rgba(0,0,0,0)'),
                    hovertext=edge_hover_text,
                    showlegend=False
                )

                # Total confidence of outgoing connections (not average)
                node_color = [
                    rules[rules['Module A'] == node]['Confidence'].sum() if node in rules['Module A'].values else 0
                    for node in G.nodes()
                ]

                node_hover_text = []
                for node in G.nodes():
                    total_conf = rules[rules['Module A'] == node]['Confidence'].sum()
                    node_hover_text.append(f"Module: {node}<br>Total Confidence: {total_conf:.3f}")

                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    text=[str(n) for n in G.nodes()],
                    textposition="bottom center",
                    hoverinfo='text',
                    marker=dict(
                        showscale=True,
                        colorscale='YlGnBu',
                        color=node_color,
                        size=20,
                        colorbar=dict(title=dict(text='Total Confidence', side='right')),
                        line_width=2
                    ),
                    hovertext=node_hover_text
                )

                fig = go.Figure(
                    data=[edge_trace, edge_hover_trace, node_trace],
                    layout=go.Layout(
                        title="Module Association Rule Visualization",
                        title_x=0.4,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    )
                )

                st.plotly_chart(fig, use_container_width=True)
