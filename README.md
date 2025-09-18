# mall_cutomers
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

# ===============================
# Step 1: Load Dataset
# ===============================
df = pd.read_csv("Mall_Customers.csv")

# ===============================
# Step 2: Exploratory Visualizations
# ===============================
# Save plots as images for embedding in PDF
plt.figure(figsize=(5,4))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title("Age Distribution")
plt.savefig("age_distribution.png")
plt.close()

plt.figure(figsize=(5,4))
sns.histplot(df['Annual Income (k$)'], bins=20, kde=True, color="green")
plt.title("Annual Income Distribution")
plt.savefig("income_distribution.png")
plt.close()

plt.figure(figsize=(5,4))
sns.histplot(df['Spending Score (1-100)'], bins=20, kde=True, color="red")
plt.title("Spending Score Distribution")
plt.savefig("spending_distribution.png")
plt.close()

# ===============================
# Step 3: KMeans Clustering
# ===============================
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(6,5))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)',
                hue='Cluster', palette='Set1', s=80)
plt.title("Customer Segments")
plt.savefig("customer_segments.png")
plt.close()

# ===============================
# Step 4: Generate PDF Report
# ===============================
doc = SimpleDocTemplate("Customer_Segmentation_Project_with_Visuals.pdf", pagesize=A4)
styles = getSampleStyleSheet()
story = []

# Title
story.append(Paragraph("Customer Segmentation Project Report", styles["Heading1"]))
story.append(Spacer(1, 20))

# Sections with analysis
sections = [
    ("Introduction",
     "Customer segmentation divides customers into groups with similar characteristics. "
     "This project applies KMeans clustering to the Mall Customers dataset to identify actionable customer groups."),

    ("Dataset",
     "The dataset has 200 customers with the following attributes: CustomerID, Gender, Age, "
     "Annual Income, and Spending Score."),

    ("Methodology",
     "1. Data exploration and visualization\n"
     "2. Feature selection: Annual Income & Spending Score\n"
     "3. Standardization\n"
     "4. KMeans clustering (k=5)\n"
     "5. Cluster profiling and visualization"),

    ("Results",
     "Cluster 0: High income, high spending (Premium Customers)\n"
     "Cluster 1: High income, low spending (Careful Customers)\n"
     "Cluster 2: Low income, high spending (Growth Potential)\n"
     "Cluster 3: Low income, low spending (Budget Customers)\n"
     "Cluster 4: Medium income, average spending (Mid-tier Customers)"),

    ("Conclusion",
     "The segmentation helps businesses target different groups:\n"
     "- Premium customers can be retained with loyalty programs.\n"
     "- Careful customers can be encouraged with personalized offers.\n"
     "- Budget customers can be engaged with discounts.\n"
     "This project shows how clustering provides actionable insights for marketing.")
]

for heading, content in sections:
    story.append(Paragraph(heading, styles["Heading2"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(content.replace("\n", "<br/>"), styles["Normal"]))
    story.append(Spacer(1, 12))

# ===============================
# Step 5: Add Visuals
# ===============================
story.append(Paragraph("Data Visualizations", styles["Heading2"]))
story.append(Spacer(1, 10))

for img in ["age_distribution.png", "income_distribution.png", 
            "spending_distribution.png", "customer_segments.png"]:
    story.append(Image(img, width=400, height=300))
    story.append(Spacer(1, 12))

# Build PDF
doc.build(story)

print("✅ PDF generated: Customer_Segmentation_Project_with_Visuals.pdf")
