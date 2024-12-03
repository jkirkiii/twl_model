import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from twl_model import TWLModel, WaterLevelDataset, EarlyStopping


def prepare_data(X, y, train_size=0.8, val_size=0.1, batch_size=32):
    """Prepare data for training with proper scaling and splitting"""
    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=train_size, random_state=42
    )

    val_size_adjusted = val_size / (1 - train_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, train_size=val_size_adjusted, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Create data loaders
    train_dataset = WaterLevelDataset(X_train_scaled, y_train)
    val_dataset = WaterLevelDataset(X_val_scaled, y_val)
    test_dataset = WaterLevelDataset(X_test_scaled, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader, scaler


def train_model(model, train_loader, val_loader, num_epochs=1500,
                learning_rate=0.01, device=None):
    """Train the neural network with early stopping and learning rate scheduling"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=200, min_lr=1e-6
    )
    early_stopping = EarlyStopping()

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            features = batch['features'].to(device)
            targets = batch['target'].to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                targets = batch['target'].to(device)
                outputs = model(features)
                val_loss += criterion(outputs, targets).item()

        # Calculate average losses
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    return train_losses, val_losses


def evaluate_model(model, test_loader, device=None):
    """Evaluate the model on test data and return performance metrics"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    criterion = nn.MSELoss()
    total_loss = 0.0
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            targets = batch['target'].to(device)
            outputs = model(features)

            total_loss += criterion(outputs, targets).item()
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    rmse = np.sqrt(avg_loss)

    return {
        'test_loss': avg_loss,
        'rmse': rmse,
        'predictions': np.array(predictions),
        'actuals': np.array(actuals)
    }


if __name__ == "__main__":
    # Example usage with dummy data
    num_samples = 1000
    input_size = 25
    output_size = 1000

    # Generate dummy data
    X = np.random.randn(num_samples, input_size)
    y = np.random.randn(num_samples, output_size)

    # Prepare data
    train_loader, val_loader, test_loader, scaler = prepare_data(X, y)

    # Initialize model
    model = TWLModel(input_size=input_size, output_size=output_size)

    # Train model
    train_losses, val_losses = train_model(model, train_loader, val_loader)

    # Evaluate model
    results = evaluate_model(model, test_loader)
    print(f"Test RMSE: {results['rmse']:.4f}")